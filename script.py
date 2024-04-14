import click

from src.main import main as main_func


@click.command()
@click.argument('mode', type=click.Choice(['train', 'inference']))
@click.argument('run_name', type=str, default=None)
@click.option('--model_name', type=click.Choice(['BaseModel', 'eff_v2_l','vit_b_16']))
@click.option('--exp_path', type=click.Path(exists=True), default='./exps/')
@click.option('--train_csv_path', type=click.Path(exists=True), default='./datas/train.csv')
@click.option('--test_csv_path', type=click.Path(exists=True), default='./datas/test.csv')
@click.option('--project_name', type=str, default='low_res_bird_img_classification')
@click.option('--wandb_logging', type=bool, default=True)
@click.option('--wandb_account_entity', type=str, default='hero981001')
@click.option('--keep_train', type=bool, default=False)
@click.option('--keep_train_model_file', type=str, default=None)
@click.option('--seed', type=int, default=456)
@click.option('--test_split_ratio', type=float, default=0.3)
@click.option('--batch_size', type=int, default='64')
@click.option('--img_resize_size', type=int, default='224')
@click.option('--lr', type=float, default=0.0001)
@click.option('--num_epochs', type=int, default=5)
@click.option('--epochs_per_save', type=int, default=5)
@click.option('--shuffle', type=bool, default=False)
@click.option('--load_model', type=str, default=None)
@click.option('--sample_submit_file_path', type=click.Path(), default = './datas/sample_submission.csv')
def script(mode, exp_path, model_name, train_csv_path, test_csv_path, project_name, wandb_logging, wandb_account_entity, run_name, seed, test_split_ratio, batch_size,
         img_resize_size, lr, num_epochs, epochs_per_save, shuffle, load_model, sample_submit_file_path, keep_train, keep_train_model_file):
    main_func(mode, exp_path, model_name, train_csv_path, test_csv_path, project_name, wandb_logging, wandb_account_entity, run_name, seed, test_split_ratio, batch_size,
         img_resize_size, lr, num_epochs, epochs_per_save, shuffle, load_model, sample_submit_file_path, keep_train, keep_train_model_file)

if __name__=="__main__":
    script()