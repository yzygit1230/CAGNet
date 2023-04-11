import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data

from models.main_model import EnsembleModel
from util.dataloaders import get_eval_loaders
from util.common import check_eval_dirs, compute_metrics, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
running_metrics =  RunningMetrics(2)
def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_info()   # 打印GPU信息
    save_path, result_save_path = check_eval_dirs()

    save_results = SaveResult(result_save_path)
    save_results.prepare()

    model = EnsembleModel(opt.ckp_paths, device, input_size=opt.input_size)

    eval_loader = get_eval_loaders(opt)

    p, r, f1, miou, oa, avg_loss = eval_for_metric(model, eval_loader, tta=opt.tta)

    # 写日志
    save_results.show(p, r, f1, miou, oa)
    print("F1-mean: {}".format(f1.mean()))
    print("mIOU-mean: {}".format(miou.mean()))


def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=448):   # 评估,得到指标(metric)
    avg_loss = 0
    val_loss = torch.tensor([0])
    scale = ScaleInOutput(input_size)

    tn_fp_fn_tp = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]

    model.eval()
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader)
        for i, (batch_img1, batch_img2, batch_label1, _) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...eval_loss: {}".format(avg_loss))
            batch_img1 = batch_img1.float().cuda()
            batch_img2 = batch_img2.float().cuda()
            batch_label1 = batch_label1.long().cuda()

            if criterion is not None:
                batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))

            outs = model(batch_img1, batch_img2, tta)

            if not isinstance(outs, tuple):
                outs = (outs, outs)
            labels = (batch_label1, batch_label1)

            if criterion is not None:
                outs = scale.scale_output(outs)
                val_loss = criterion(outs, labels)  # ((B,2 H,W), (B,2 H,W)) and ((B,H,W), (B,H,W))
                _, cd_pred1 = torch.max(outs[0], 1)   # 二值化预测结果
                _, cd_pred2 = torch.max(outs[1], 1)
            else:
                cd_pred1 = outs[0]
                cd_pred2 = outs[1]

            cd_preds = (cd_pred1, cd_pred2)
            running_metrics.update(labels[0].data.cpu().numpy(),cd_preds[0].data.cpu().numpy())
            avg_loss = (avg_loss * i + val_loss.cpu().detach().numpy()) / (i + 1)

            for j, (cd_pred, label) in enumerate(zip(cd_preds, labels)):
                tn = ((cd_pred == 0) & (label == 0)).int().sum().cpu().numpy()
                fp = ((cd_pred == 1) & (label == 0)).int().sum().cpu().numpy()
                fn = ((cd_pred == 0) & (label == 1)).int().sum().cpu().numpy()
                tp = ((cd_pred == 1) & (label == 1)).int().sum().cpu().numpy()
                assert tn+fp+fn+tp == np.prod(batch_label1.shape)

                tn_fp_fn_tp[j] += [tn, fp, fn, tp]

    p, r, f1, miou, iou_0, iou_1, oa, kappa = compute_metrics(tn_fp_fn_tp)
    score = running_metrics.get_scores()
    return p, r, f1, miou, oa, avg_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection eval')
    parser.add_argument("--ckp-paths", type=str,
                        default=[
                            "weight-path",
                        ])
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--dataset-dir", type=str, default="dataset-path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--tta", type=bool, default=False)

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)
