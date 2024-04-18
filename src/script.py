import click
import importlib

@click.group()
def cli():
    pass

@cli.command()
@click.argument('run_name', type=str, default=None)
@click.option('--model_name', type=click.Choice(['BaseModel', 'eff_v2_l','eff_b7','vit_b_16','swin_v2']))
@click.option('--exp_path', type=click.Path(exists=True), default='../exps/')
@click.option('--project_name', type=str, default='low_res_bird_img_classification')
@click.option('--seed', type=int, default=456)
@click.option('--batch_size', type=int, default='64')
@click.option('--img_resize_size', type=int, default='224')
@click.option('--shuffle', type=bool, default=False)
@click.option('--train_csv_path', type=click.Path(exists=True), default='../datas/train.csv')
@click.option('--wandb_logging', type=bool, default=True)
@click.option('--wandb_account_entity', type=str, default='hero981001')
@click.option('--keep_train', type=bool, default=False)
@click.option('--keep_train_model_file', type=str, default=None)
@click.option('--test_split_ratio', type=float, default=0.3)
@click.option('--lr', type=float, default=0.0001)
@click.option('--num_epochs', type=int, default=5)
@click.option('--epochs_per_save', type=int, default=5)
def train(run_name, model_name, exp_path, 
          project_name, seed, batch_size, 
          img_resize_size, shuffle, train_csv_path, 
          wandb_logging, wandb_account_entity, keep_train, keep_train_model_file, 
          test_split_ratio, lr, num_epochs, epochs_per_save):
    train_module = importlib.import_module("functions")
    train_func = getattr(train_module, 'train_func')
    train_func(run_name, model_name, exp_path, 
               project_name, seed, batch_size, 
               img_resize_size, shuffle, train_csv_path, 
               wandb_logging, wandb_account_entity, keep_train, keep_train_model_file, 
               test_split_ratio, lr, num_epochs, epochs_per_save)


@cli.command()
@click.argument('run_name', type=str, default=None)
@click.option('--model_name', type=click.Choice(['BaseModel', 'eff_v2_l','eff_b7','vit_b_16','swin_v2']))
@click.option('--exp_path', type=click.Path(exists=True), default='../exps/')
@click.option('--project_name', type=str, default='low_res_bird_img_classification')
@click.option('--seed', type=int, default=456)
@click.option('--batch_size', type=int, default='64')
@click.option('--img_resize_size', type=int, default='224')
@click.option('--shuffle', type=bool, default=False)
@click.option('--test_csv_path', type=click.Path(exists=True), default='../datas/test.csv')
@click.option('--load_model', type=str, default=None)
@click.option('--sample_submit_file_path', type=click.Path(), default = '../datas/sample_submission.csv')
def inference(run_name, model_name, exp_path, 
              project_name, seed, batch_size, 
              img_resize_size, shuffle, test_csv_path,
              load_model, sample_submit_file_path):
    inference_module = importlib.import_module("functions")
    inference_func = getattr(inference_module, 'inference_func')
    inference_func(run_name, model_name, exp_path, 
                   project_name, seed, batch_size, 
                   img_resize_size, shuffle, test_csv_path,
                   load_model, sample_submit_file_path)


if __name__ == '__main__':
    cli()
    