import math
import os
import ssl
import time

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image as imwrite

from perceptual import PerceptLoss

from Model import Base_Model, Discriminator
from Model_util import padding_image, parser, Lap_Pyramid_Conv
from make import getTxt
from Loss import SSIMLoss
from test_dataset import dehaze_test_dataset
from train_dataset import dehaze_train_dataset
from utils_test import to_psnr, to_ssim_skimage

ssl._create_default_https_context = ssl._create_unverified_context

args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
start_epoch = 0
sep = args.sep

tag = 'else'
if args.type == 0:
    args.train_dir = '../datasets_train/thick_660/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Haze1k/Haze1k_thin/dataset/test/"
    args.test_name = 'input,target'
    tag = 'thin'
elif args.type == 1:
    args.train_dir = '../datasets_train/thick_660/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Haze1k/Haze1k_moderate/dataset/test/"
    args.test_name = 'input,target'
    tag = 'moderate'
elif args.type == 2:
    args.train_dir = '../datasets_train/thick_train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Haze1k/Haze1k_thick/dataset/test/"
    args.test_name = 'input,target'
    tag = 'thick'
elif args.type == 3:
    args.train_dir = '../datasets_train/Dense_hazy/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Dense_hazy/test/"
    args.test_name = 'hazy,clean'
    tag = 'dense'
elif args.type == 4:
    args.train_dir = '../datasets_train/nhhaze/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/nhhaze/test/"
    args.test_name = 'hazy,clean'
    tag = 'nhhaze'
elif args.type == 5:
    args.train_dir = '../datasets_train/Outdoor/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/Outdoor/test/"
    args.test_name = 'hazy,clean'
    tag = 'outdoor'
elif args.type == 6:
    args.train_dir = '../datasets_train/rice1/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/rice1/"
    args.test_name = 'hazy,clean'
    tag = 'rice1'
elif args.type == 7:
    args.train_dir = '../datasets_train/rice2/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "../datasets_test/rice2/"
    args.test_name = 'hazy,clean'
    tag = 'rice2'
elif args.type == 8:
    args.train_dir = '../datasets_train/thin_cloudy/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/thin_cloudy/"
    args.test_name = 'hazy,clear'
    tag = 'thin_cloudy'
elif args.type == 9:
    args.train_dir = '../datasets_train/LHID/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/LHID/"
    args.test_name = 'hazy,clear'
    tag = 'LHID'
elif args.type == 10:
    args.train_dir = '../datasets_train/DHID/train/'
    args.train_name = 'hazy,clear'
    args.test_dir = "../datasets_test/DHID/"
    args.test_name = 'hazy,clear'
    tag = 'DHID'
elif args.type == 11:
    args.train_dir = '../datasets_train/indoor/train/'
    args.train_name = 'hazy,gt'
    args.test_dir = "../datasets_test/indoor/"
    args.test_name = 'hazy,gt'
    tag = 'indoor'


print('We are training datasets: ', tag)

getTxt(args.train_dir, args.train_name, args.test_dir, args.test_name)

predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
# output_dir = os.path.join(args.model_save_dir, 'output_result')

# --- Gpu device --- #
device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]

print('use gpus ->', args.gpus)
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
if args.use_bn:
    print('we are using BatchNorm')
else:
    print('we are using InstanceNorm')

'''SDN = Base_Model(bn=args.use_bn)
print('SDN parameters:', sum(param.numel() for param in SDN.parameters()))
DNet = Discriminator(bn=args.use_bn)
print('Discriminator parameters:', sum(param.numel() for param in DNet.parameters()))'''

# --- Build optimizer --- #
'''G_optimizer = torch.optim.Adam(SDN.parameters(), lr=0.0001)
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[5000, 7000, 8000], gamma=0.5)
D_optim = torch.optim.Adam(DNet.parameters(), lr=0.0001)
scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optim, milestones=[5000, 7000, 8000], gamma=0.5)
'''

generator = Base_Model()
discriminator = Discriminator()  # 输出 bz 1 30 30
discriminator1 = Discriminator()

criterionSsim = SSIMLoss()
criterion = torch.nn.MSELoss()
criterionP = torch.nn.L1Loss()
criterionC = PerceptLoss(device=device)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
optimizer_T = torch.optim.Adam([
    {'params': generator.parameters(), 'lr': 0.0001}
])

generator.to(device)
criterion.to(device)
criterionP.to(device)
discriminator.to(device)
discriminator1.to(device)
criterionSsim.to(device)

scheduler_T = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_T, T_max=args.train_epoch, eta_min=0, last_epoch=-1)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.train_epoch, eta_min=0, last_epoch=-1)

# --- Load training data --- #
dataset = dehaze_train_dataset(args.train_dir, args.train_name, tag)
print('trainDataset len: ', len(dataset))
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, drop_last=True,
                          num_workers=4)
# --- Load testing data --- #

test_dataset = dehaze_test_dataset(args.test_dir, args.test_name, tag)
print('testDataset len: ', len(test_dataset))
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0,
                         pin_memory=True)

# val_dataset = dehaze_val_dataset(val_dataset)
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

laplace = Lap_Pyramid_Conv(num_high=1, device=device).to(device)

# --- Multi-GPU --- #
'''SDN = SDN.to(device)
SDN = torch.nn.DataParallel(SDN, device_ids=device_ids)
DNet = DNet.to(device)
DNet = torch.nn.DataParallel(DNet, device_ids=device_ids)'''
writer = SummaryWriter(os.path.join(args.model_save_dir, 'tensorboard'))

# --- Define the perceptual loss network --- #

# --- Load the network weight --- #
if args.restart:
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
    name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
    generator.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(num))
    start_epoch = int(num) + 1
elif args.num != '9999999':
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
    generator.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(args.num))
    start_epoch = int(args.num) + 1
else:
    print('--- no weight loaded ---')

iteration = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
pl = []
sl = []
best_psnr = 0
best_psnr_ssim = 0
best_ssim = 0
best_ssim_psnr = 0
print()
start_time = time.time()

for epoch in range(start_epoch, train_epoch):
    print('++{}+++++++++{}+++++++++++++ {} Datasets +++++++ {} epoch ++++++++++++++++++++++++'.format(
        'gpus:' + str(device), os.getcwd().split("/")[-1], tag, epoch))
    '''scheduler_G.step()
    scheduler_D.step()'''
    generator.train()
    discriminator.train()
    with tqdm(total=len(train_loader)) as t:
        for (x, y) in train_loader:
            # print(batch_idx)

            h_list = [i for i in laplace.pyramid_decom(x)]
            # c_list = [i for i in laplace.pyramid_decom(clean)]
            iteration += 1

            x = x.to(device)
            y = y.to(device)
            real_label = torch.ones((x.size()[0], 1, 30, 30), requires_grad=False).to(device)
            fake_label = torch.zeros((x.size()[0], 1, 30, 30), requires_grad=False).to(device)

            real_out = discriminator(y)
            real_out1 = discriminator1(y)

            loss_real_D = criterion(real_out, real_label)
            loss_real_D1 = criterion(real_out1, real_label)

            p_haze, p_dehaze, fake_img = generator(x, h_list[0])

            fake_out = discriminator(fake_img.detach())
            p_de_out = discriminator1(p_dehaze.detach())

            loss_fake_D = criterion(fake_out, fake_label)
            loss_fake_D1 = criterion(p_de_out, fake_label)

            loss_D = (loss_real_D + loss_fake_D + loss_fake_D1) / 3

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            fake_img_ = fake_img
            haze = p_haze
            dehaze = p_dehaze

            output = discriminator(fake_img_)

            de_output = discriminator1(dehaze)

            loss_G = criterion(output, real_label)
            loss_G1 = criterion(de_output, real_label)
                                               #  风格损失
            loss_P = criterionP(haze, x) # + 0.5 * criterionSsim(haze, x) # L1             #  大气散射模型约束

            loss_Right = criterionP(fake_img_, dehaze.detach())  # 右拉
            # loss_ssim = criterionSsim(fake_img_, dehaze.detach())  # 结构
            loss_C1 = criterionC(fake_img_, dehaze.detach())  # 对比下
            loss_C2 = criterionC(haze, x,)  # 对比上

            total_loss = loss_G + loss_P + loss_Right + loss_C1 + loss_C2

            optimizer_T.zero_grad()
            total_loss.backward()
            optimizer_T.step()

            '''DNet.zero_grad()

            real_out = DNet(clean_256).mean()
            fake_out = DNet(img_256.detach()).mean()

            D_loss = 1 - real_out + fake_out

            D_loss.backward(retain_graph=True)

            SDN.zero_grad()

            fake_out = DNet(img_256).mean()
            adversarial_loss_256 = torch.mean(1 - fake_out)
            smooth_loss_l1_256 = F.smooth_l1_loss(img_256, haze.detach())
            perceptual_loss_256 = loss_network(img_256, haze.detach())
            msssim_loss_256 = -msssim_loss(img_256, haze.detach(), normalize=True)

            # total_64 = smooth_loss_l1_64 + 0.01 * perceptual_loss_64 + 0.0005 * adversarial_loss_64 + 0.5 * msssim_loss_64
            # total_128 = smooth_loss_l1_128 + 0.01 * perceptual_loss_128 + 0.0005 * adversarial_loss_128 + 0.5 * msssim_loss_128
            total_256 = smooth_loss_l1_256 + 0.01 * perceptual_loss_256 + 0.0005 * adversarial_loss_256 + 0.5 * msssim_loss_256

            total_loss = total_256  # 0.01 * perceptual_loss_img + 0.0005 * adversarial_loss_img + 0.5 * msssim_loss_img +

            total_loss.backward()
            D_optim.step()
            G_optimizer.step()'''

            #         if iteration % 2 == 0:
            #             frame_debug = torch.cat(
            #                 (hazy, output, clean), dim=0)
            #             writer.add_images('train_debug_img', frame_debug, iteration)
            '''writer.add_scalars('training', {'training total loss': total_loss.item()
                                            }, iteration)
            writer.add_scalars('training_img', {'img loss_l1': smooth_loss_l1_256.item(),
                                                'perceptual': perceptual_loss_256.item(),
                                                'msssim': msssim_loss_256.item()

                                                }, iteration)
            writer.add_scalars('GAN_training', {
                'd_loss': D_loss.item(),
                'd_score': real_out.item(),
                'g_score': fake_out.item()
            }, iteration
                               )'''
            step_2_loss = loss_G
            step_1_loss = loss_P
            co_loss = loss_Right
            t.set_description(
                "===> Epoch[{}] : step_1_loss: {:.2f} step_2_loss: {:.2f} co_loss: {:.2f} G_loss: {:.2f} D_loss: {:.2f} ".format(
                    epoch, step_1_loss.item(), step_2_loss.item(), co_loss.item(), total_loss.item(), loss_C1.item(),
                    time.time() - start_time))
            t.update(1)

    if args.seps:
        torch.save(generator.state_dict(),
                   os.path.join(args.model_save_dir,
                                'epoch_' + str(epoch) + '_' + '.pkl'))
        continue

    if tag in ['outdoor']:
        if epoch >= 20:
            sep = 1
    elif tag in ['thin', 'thick', 'moderate']:
        if epoch >= 100:
            sep = 1
    else:
        if epoch >= 500:
            sep = 1

    if epoch % sep == 0:

        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            generator.eval()
            for (hazy, clean, name) in tqdm(test_loader):
                hazy = hazy.to(device)
                clean = clean.to(device)

                h_list = [i for i in laplace.pyramid_decom(hazy)]
                high = h_list[0]

                h, w = hazy.shape[2], hazy.shape[3]
                max_h = int(math.ceil(h / 256)) * 256
                max_w = int(math.ceil(w / 256)) * 256
                hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)
                high, _, _, _, _ = padding_image(high, max_h, max_w)
                haze, dehaze, img = generator(hazy, high)

                img = img.data[:, :, ori_top:ori_down, ori_left:ori_right]
                dehaze = dehaze[:, :, ori_top:ori_down, ori_left:ori_right]

                # frame_out_up = SDN(hazy_up)
                # frame_out_down = SDN(hazy_down)
                # frame_out=(torch.cat([frame_out_up.permute(0,2,3,1), frame_out_down[:,:,80:640,:].permute(0,2,3,1)], 1)).permute(0,3,1,2)

                #                 imwrite(frame_out, output_dir +'/' +str(batch_idx) + '.png', range=(0, 1))
                uresult = to_psnr(img, clean)
                atpresult = to_psnr(dehaze, clean)



                if atpresult >= uresult:
                    psnr_list.extend(to_psnr(dehaze, clean))
                    ssim_list.extend(to_ssim_skimage(dehaze, clean))

                else:
                    psnr_list.extend(to_psnr(img, clean))
                    ssim_list.extend(to_ssim_skimage(img, clean))


            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            pl.append(avr_psnr)
            sl.append(avr_ssim)

            if avr_psnr >= max(pl):
                best_epoch_psnr = epoch
                best_psnr = avr_psnr
                best_psnr_ssim = avr_ssim

            if avr_ssim >= max(sl):
                best_epoch_ssim = epoch
                best_ssim = avr_ssim
                best_ssim_psnr = avr_psnr


            print(epoch, 'dehazed', avr_psnr, avr_ssim)
            if best_epoch_psnr == best_epoch_ssim:
                print('best epoch is {}, psnr: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_ssim))
            else:
                print('best psnr epoch is {}: PSNR: {}, ssim: {}'.format(best_epoch_psnr, best_psnr, best_psnr_ssim))
                print('best ssim epoch is {}: psnr: {}, SSIM: {}'.format(best_epoch_ssim, best_ssim_psnr, best_ssim))
            print()
            frame_debug = torch.cat((img, clean), dim=0)
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr': avr_psnr,
                                           'testing ssim': avr_ssim
                                           }, epoch)
            if best_epoch_psnr == epoch or best_epoch_ssim == epoch:
                torch.save(generator.state_dict(),
                           os.path.join(args.model_save_dir,
                                        'epoch_' + str(epoch) + '_' + str(round(avr_psnr, 2)) + '_' + str(
                                            round(avr_ssim, 3)) + '_' + str(tag) + '.pkl'))


os.remove(os.path.join(args.train_dir, 'train.txt'))
os.remove(os.path.join(args.test_dir, 'test.txt'))
writer.close()
