import importlib
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TestSetDataLoader, MultiValSetDataLoader
from collections import OrderedDict
from train import test
import random


def main(args):
    ''' Create Dir for Save '''
    _, result_dir, _ = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    Test_Dataset = TestSetDataLoader(args)
    Test_Loaders = torch.utils.data.DataLoader(dataset=Test_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=False)
    test_Names, test_Loaders, length_of_tests = MultiValSetDataLoader(args)
    print("The number of test data is: %d" % len(Test_Loaders))

    test_name = 'Test'

    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        pass
    else:
        ckpt_path = args.path_pre_pth
        checkpoint = torch.load(ckpt_path, map_location='cuda:0')
        try:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = 'module.' + k  # add `module.`
                new_state_dict[name] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
        except:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k] = v
            # load params
            net.load_state_dict(new_state_dict)
            print('Use pretrain model!')
            pass
        pass

    # net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''
        excel_file = ExcelFile()
        excel_file_allviews = ExcelFile_allviews()
        psnr_testset = []
        ssim_testset = []

        save_dir = result_dir.joinpath(test_name)
        save_dir.mkdir(exist_ok=True)

        CenterView_dir = save_dir.joinpath('CenterView/')
        CenterView_dir.mkdir(exist_ok=True)

        mat_file = save_dir.joinpath('mat_file/')
        mat_file.mkdir(exist_ok=True)

        LF_epoch_save_dir = save_dir.joinpath('Results/')
        LF_epoch_save_dir.mkdir(exist_ok=True)

        aligned_image_dir = save_dir.joinpath('aligned_image')
        aligned_image_dir.mkdir(exist_ok=True)

        psnr_iter_test, ssim_iter_test, LF_name, psnr_iter_test_allviews,ssim_iter_test_allviews, single_sence_time_avg = test(Test_Loaders, device, net, save_dir,
                                                                        CenterView_dir, mat_file, aligned_image_dir)
        excel_file.write_sheet(test_name, LF_name, psnr_iter_test, ssim_iter_test)
        excel_file_allviews.write_sheet(test_Names,LF_name, psnr_iter_test, ssim_iter_test,psnr_iter_test_allviews,ssim_iter_test_allviews)

        psnr_epoch_test = float(np.array(psnr_iter_test).mean())
        ssim_epoch_test = float(np.array(ssim_iter_test).mean())
        psnr_testset.append(psnr_epoch_test)
        ssim_testset.append(ssim_epoch_test)
        print('Test psnr/ssim is %.2f/%.3f' % (psnr_epoch_test, ssim_epoch_test))
        pass

        psnr_mean_test = float(np.array(psnr_testset).mean())
        ssim_mean_test = float(np.array(ssim_testset).mean())
        excel_file.add_sheet('ALL', 'Average', psnr_mean_test, ssim_mean_test)
        excel_file_allviews.add_sheet('ALL', 'Average', psnr_mean_test, ssim_mean_test)
        print('The mean psnr on testsets is %.5f, mean ssim is %.5f' % (psnr_mean_test, ssim_mean_test))
        excel_file.xlsx_file.save(str(result_dir) + '/evaluation.xls')
        excel_file_allviews.xlsx_file.save(str(result_dir) + '/evaluation_allviews.xls')
    pass

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    from config import args

    main(args)
