import argparse
import torch
from utils.datautils import load_forecast_csv
from train import train
from tasks.forecasting import predict
from utils.otherutils import save_model, load_model, save_metric, load_metric

parser = argparse.ArgumentParser()
'''
Dataset
'''
parser.add_argument('--name', type=str, default='exchange_rate', help='The datasets name')
# parser.add_argument('--name', type=str, default='ETTh1', help='The datasets name')
# parser.add_argument('--name', type=str, default='ETTh2', help='The datasets name')
# parser.add_argument('--name', type=str, default='ETTm1', help='The datasets name')
# parser.add_argument('--name', type=str, default='weather', help='The datasets name')

'''
Route
'''
parser.add_argument('--saved_data', type=str, default='datasets', help='The dir to save source datasets')
parser.add_argument('--saved_model', type=str, default='saved_model', help='The dir to save model')
parser.add_argument('--saved_plot', type=str, default='saved_plot', help='The dir to save plot')
parser.add_argument('--saved_metric', type=str, default='saved_metric', help='The dir to save metric')
parser.add_argument('--archive', type=str, default='forecast_csv',
                    help='Choose load mode(forecast_csv or forecast_csv_npy)')

'''
Training task 1
CycleGAN hyper-parameters
'''
parser.add_argument('--lr_g', type=float, default=1e-3, help='Learning rate about generator')
parser.add_argument('--lr_d', type=float, default=1e-6, help='Learning  rate about discriminator')
parser.add_argument('--train_gan_epochs', type=int, default=50, help='Epochs of GAN training in task 1')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--similarity', type=float, default=1.0, help='Similarity about 2 samples')

'''
Training task 2
Contrastive learning hyper-parameters
'''
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--lr', type=float, default=6.25e-3, help='Learning Rate')
parser.add_argument('--epochs', type=int, default=50, help='Num of train epochs')
parser.add_argument('--repr_dims', type=int, default=320, help='Representation dimensions')
parser.add_argument('--hidden_dims', type=int, default=64, help='Hidden dimensions in dilated conv')
parser.add_argument('--depth', type=int, default=10, help='Depth of dilated conv')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to Train')
parser.add_argument('--univar', type=bool, default=False, help='Whether to predict only one feature')
parser.add_argument('--univar_feature', type=str, default=None, help='Which feature to predict')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Training ratio')
parser.add_argument('--valid_ratio', type=float, default=0.2, help='Validation ratio')
parser.add_argument('--test_ratio', type=float, default=0.2, help='Testing ratio')
parser.add_argument('--max_train_length', type=int, default=1000, help='Max training length')
parser.add_argument('--kernels', type=list, default=[2, 4, 6, 8, 10])
parser.add_argument('--alpha', type=float, default=0.0005)
parser.add_argument('--length_queue', type=int, default=256, help='Length of queue')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--pred_lens', type=list, default=[24, 48, 168, 336, 720], help='Prediction length')
parser.add_argument('--downstream_mode', type=str, default='forecasting', help='Downstream task type')

'''
Others
'''
parser.add_argument('--isEval', type=bool, default=True, help='Whether eval or not')
parser.add_argument('--isTrain', type=bool, default=True, help='Whether train or use saved model')

args = parser.parse_args()

'''
Main function
'''
if __name__ == '__main__':
    load_data_config = dict(
        saved_data=args.saved_data,
        name=args.name,
        univar=args.univar,
        univar_feature=args.univar_feature,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )
    if args.archive == 'forecast_csv':
        data, train_slice, valid_slice, test_slice, scaler, n_covariant_cols, col_names = load_forecast_csv(
            config=load_data_config)
        train_data = data[:, train_slice]
        pass
    elif args.archive == 'forecast_csv_npy':
        pass
    else:
        raise ValueError(f'Archive type {args.archive} is not supported')

    training_gan_config = dict(
        dropout=args.dropout,
        device=args.device,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        epochs=args.train_gan_epochs
    )

    training_config = dict(
        train_data=train_data,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        output_dims=args.repr_dims,
        input_dims=train_data.shape[-1],
        kernels=args.kernels,
        alpha=args.alpha,
        max_train_length=args.max_train_length,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        length_queue=args.length_queue,
        num_workers=args.num_workers,
        mode=args.downstream_mode,
        device=args.device,
        gan_config=training_gan_config,
        similarity=args.similarity
    )

    # Train the model if necessary
    if args.isTrain:
        model, loss_log = train(config=training_config)
        save_model(save_path=args.saved_model, model=model, dataset=args.name, epochs=args.epochs)
        pass
    else:
        model = load_model(save_path=args.saved_model, dataset=args.name, epochs=args.epochs)
        pass

    forecasting_config = dict(
        model=model,
        data=data,
        train_slice=train_slice,
        valid_slice=valid_slice,
        test_slice=test_slice,
        scaler=scaler,
        pred_lens=args.pred_lens,
        n_covariant_cols=n_covariant_cols,
        padding=args.max_train_length - 1,
        mode=args.downstream_mode,
        device=args.device,
        name=args.name
    )
    result_metric = predict(config=forecasting_config)

    for step in args.pred_lens:
        result = result_metric[step]['raw']
        save_metric(save_path=args.saved_metric, dataset=args.name, epochs=args.epochs, metric=result,
                    current_pred_len=step)
        pass

    for step in args.pred_lens:
        MSE, MAE = load_metric(save_path=args.saved_metric, dataset=args.name, epochs=args.epochs,
                               current_pred_len=step)
        print(f'horizon = {step}  MSE: {MSE}, MAE: {MAE}')
        pass
    pass
