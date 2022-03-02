# -*- coding: utf-8 -*-
#This code was built on the original code written by Mazziar Raissi: https://github.com/maziarraissi/PINNs

#author: Gokul Subraveti

import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class PANACHE:
    #Initialize the class
    def __init__(self, X0, X_en, X_lb, X_rb, in_train, en_train, lb_train, 
                 rb_train, X_c, layers, low_bound, up_bound, coeff, N0, N_b, N_c, N_k):
        
        self.low_bound = low_bound
        self.up_bound = up_bound

        self.N0 = N0
        self.N_b = N_b
        self.N_c = N_c
        self.N_k = N_k
        
        self.z0 = X0[:,0:1]
        self.t0 = X0[:,1:2]
        self.m10 = X0[:,2:3]
        self.m20 = X0[:,3:4]
        self.m30 = X0[:,4:5]
        self.m40 = X0[:,5:6]
        self.m50 = X0[:,6:7]
        self.m60 = X0[:,7:8]
        self.m70 = X0[:,8:9]
        self.m80 = X0[:,9:10]
        self.m90 = X0[:,10:11]
        self.m100 = X0[:,11:12]
        self.m110 = X0[:,12:13]
        self.m120 = X0[:,13:14]
        self.m130 = X0[:,14:15] 
        self.m140 = X0[:,15:16]   
        self.m150 = X0[:,16:17] 
        self.m160 = X0[:,17:18] 
        self.m170 = X0[:,18:19] 
        self.m180 = X0[:,19:20] 
        self.m190 = X0[:,20:21]
        self.m200 = X0[:,21:22]
        self.m210 = X0[:,22:23]
        self.m220 = X0[:,23:24]
        self.m230 = X0[:,24:25] 
        self.m240 = X0[:,25:26] 
        self.m250 = X0[:,26:27] 
        self.m260 = X0[:,27:28] 
        self.m270 = X0[:,28:29] 
        self.m280 = X0[:,29:30] 
        self.m290 = X0[:,30:31]
        self.m300 = X0[:,31:32]
        self.m310 = X0[:,32:33]
        self.m320 = X0[:,33:34]
        self.m330 = X0[:,34:35] 
        self.m340 = X0[:,35:36]  
        self.m350 = X0[:,36:37]
        self.m360 = X0[:,37:38]
        self.m370 = X0[:,38:39]
        self.m380 = X0[:,39:40]
        self.m390 = X0[:,40:41]
        self.m400 = X0[:,41:42]
        self.m410 = X0[:,42:43]
        self.m420 = X0[:,43:44]
        self.m430 = X0[:,44:45] 
        self.m440 = X0[:,45:46]
        self.m450 = X0[:,46:47]
        self.m460 = X0[:,47:48]
        self.m470 = X0[:,48:49]
        self.m480 = X0[:,49:50]
        self.m490 = X0[:,50:51]
        self.m500 = X0[:,51:52]

        self.z_en = X_en[:,0:1]
        self.t_en = X_en[:,1:2]
        self.m1_en = X_en[:,2:3]
        self.m2_en = X_en[:,3:4]
        self.m3_en = X_en[:,4:5]
        self.m4_en = X_en[:,5:6]
        self.m5_en = X_en[:,6:7]
        self.m6_en = X_en[:,7:8]
        self.m7_en = X_en[:,8:9]
        self.m8_en = X_en[:,9:10]
        self.m9_en = X_en[:,10:11]
        self.m10_en = X_en[:,11:12]
        self.m11_en = X_en[:,12:13]
        self.m12_en = X_en[:,13:14]
        self.m13_en = X_en[:,14:15] 
        self.m14_en = X_en[:,15:16]   
        self.m15_en = X_en[:,16:17] 
        self.m16_en = X_en[:,17:18] 
        self.m17_en = X_en[:,18:19] 
        self.m18_en = X_en[:,19:20] 
        self.m19_en = X_en[:,20:21]
        self.m20_en = X_en[:,21:22]
        self.m21_en = X_en[:,22:23]
        self.m22_en = X_en[:,23:24]
        self.m23_en = X_en[:,24:25] 
        self.m24_en = X_en[:,25:26] 
        self.m25_en = X_en[:,26:27] 
        self.m26_en = X_en[:,27:28] 
        self.m27_en = X_en[:,28:29] 
        self.m28_en = X_en[:,29:30] 
        self.m29_en = X_en[:,30:31]
        self.m30_en = X_en[:,31:32]
        self.m31_en = X_en[:,32:33]
        self.m32_en = X_en[:,33:34]
        self.m33_en = X_en[:,34:35] 
        self.m34_en = X_en[:,35:36]  
        self.m35_en = X_en[:,36:37]
        self.m36_en = X_en[:,37:38]
        self.m37_en = X_en[:,38:39]
        self.m38_en = X_en[:,39:40]
        self.m39_en = X_en[:,40:41]
        self.m40_en = X_en[:,41:42]
        self.m41_en = X_en[:,42:43]
        self.m42_en = X_en[:,43:44]
        self.m43_en = X_en[:,44:45] 
        self.m44_en = X_en[:,45:46]
        self.m45_en = X_en[:,46:47]
        self.m46_en = X_en[:,47:48]
        self.m47_en = X_en[:,48:49]
        self.m48_en = X_en[:,49:50]
        self.m49_en = X_en[:,50:51]
        self.m50_en = X_en[:,51:52]

        self.z_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]
        self.m1_lb = X_lb[:,2:3]
        self.m2_lb = X_lb[:,3:4]
        self.m3_lb = X_lb[:,4:5]
        self.m4_lb = X_lb[:,5:6]
        self.m5_lb = X_lb[:,6:7]
        self.m6_lb = X_lb[:,7:8]
        self.m7_lb = X_lb[:,8:9]
        self.m8_lb = X_lb[:,9:10]
        self.m9_lb = X_lb[:,10:11]
        self.m10_lb = X_lb[:,11:12]
        self.m11_lb = X_lb[:,12:13]
        self.m12_lb = X_lb[:,13:14]
        self.m13_lb = X_lb[:,14:15] 
        self.m14_lb = X_lb[:,15:16]   
        self.m15_lb = X_lb[:,16:17] 
        self.m16_lb = X_lb[:,17:18] 
        self.m17_lb = X_lb[:,18:19] 
        self.m18_lb = X_lb[:,19:20] 
        self.m19_lb = X_lb[:,20:21]
        self.m20_lb = X_lb[:,21:22]
        self.m21_lb = X_lb[:,22:23]
        self.m22_lb = X_lb[:,23:24]
        self.m23_lb = X_lb[:,24:25] 
        self.m24_lb = X_lb[:,25:26] 
        self.m25_lb = X_lb[:,26:27] 
        self.m26_lb = X_lb[:,27:28] 
        self.m27_lb = X_lb[:,28:29] 
        self.m28_lb = X_lb[:,29:30] 
        self.m29_lb = X_lb[:,30:31]
        self.m30_lb = X_lb[:,31:32]
        self.m31_lb = X_lb[:,32:33]
        self.m32_lb = X_lb[:,33:34]
        self.m33_lb = X_lb[:,34:35] 
        self.m34_lb = X_lb[:,35:36]  
        self.m35_lb = X_lb[:,36:37]
        self.m36_lb = X_lb[:,37:38]
        self.m37_lb = X_lb[:,38:39]
        self.m38_lb = X_lb[:,39:40]
        self.m39_lb = X_lb[:,40:41]
        self.m40_lb = X_lb[:,41:42]
        self.m41_lb = X_lb[:,42:43]
        self.m42_lb = X_lb[:,43:44]
        self.m43_lb = X_lb[:,44:45] 
        self.m44_lb = X_lb[:,45:46]
        self.m45_lb = X_lb[:,46:47]
        self.m46_lb = X_lb[:,47:48]
        self.m47_lb = X_lb[:,48:49]
        self.m48_lb = X_lb[:,49:50]
        self.m49_lb = X_lb[:,50:51]
        self.m50_lb = X_lb[:,51:52]

        self.z_rb = X_rb[:,0:1]
        self.t_rb = X_rb[:,1:2]
        self.m1_rb = X_rb[:,2:3]
        self.m2_rb = X_rb[:,3:4]
        self.m3_rb = X_rb[:,4:5]
        self.m4_rb = X_rb[:,5:6]
        self.m5_rb = X_rb[:,6:7]
        self.m6_rb = X_rb[:,7:8]
        self.m7_rb = X_rb[:,8:9]
        self.m8_rb = X_rb[:,9:10]
        self.m9_rb = X_rb[:,10:11]
        self.m10_rb = X_rb[:,11:12]
        self.m11_rb = X_rb[:,12:13]
        self.m12_rb = X_rb[:,13:14]
        self.m13_rb = X_rb[:,14:15] 
        self.m14_rb = X_rb[:,15:16]   
        self.m15_rb = X_rb[:,16:17] 
        self.m16_rb = X_rb[:,17:18] 
        self.m17_rb = X_rb[:,18:19] 
        self.m18_rb = X_rb[:,19:20] 
        self.m19_rb = X_rb[:,20:21]
        self.m20_rb = X_rb[:,21:22]
        self.m21_rb = X_rb[:,22:23]
        self.m22_rb = X_rb[:,23:24]
        self.m23_rb = X_rb[:,24:25] 
        self.m24_rb = X_rb[:,25:26] 
        self.m25_rb = X_rb[:,26:27] 
        self.m26_rb = X_rb[:,27:28] 
        self.m27_rb = X_rb[:,28:29] 
        self.m28_rb = X_rb[:,29:30] 
        self.m29_rb = X_rb[:,30:31]
        self.m30_rb = X_rb[:,31:32]
        self.m31_rb = X_rb[:,32:33]
        self.m32_rb = X_rb[:,33:34]
        self.m33_rb = X_rb[:,34:35] 
        self.m34_rb = X_rb[:,35:36]  
        self.m35_rb = X_rb[:,36:37]
        self.m36_rb = X_rb[:,37:38]
        self.m37_rb = X_rb[:,38:39]
        self.m38_rb = X_rb[:,39:40]
        self.m39_rb = X_rb[:,40:41]
        self.m40_rb = X_rb[:,41:42]
        self.m41_rb = X_rb[:,42:43]
        self.m42_rb = X_rb[:,43:44]
        self.m43_rb = X_rb[:,44:45] 
        self.m44_rb = X_rb[:,45:46]
        self.m45_rb = X_rb[:,46:47]
        self.m46_rb = X_rb[:,47:48]
        self.m47_rb = X_rb[:,48:49]
        self.m48_rb = X_rb[:,49:50]
        self.m49_rb = X_rb[:,50:51]
        self.m50_rb = X_rb[:,51:52]

        self.z_c = X_c[:,0:1]
        self.t_c = X_c[:,1:2]
        self.m1_c = X_c[:,2:3]
        self.m2_c = X_c[:,3:4]
        self.m3_c = X_c[:,4:5]
        self.m4_c = X_c[:,5:6]
        self.m5_c = X_c[:,6:7]
        self.m6_c = X_c[:,7:8]
        self.m7_c = X_c[:,8:9]
        self.m8_c = X_c[:,9:10]
        self.m9_c = X_c[:,10:11]
        self.m10_c = X_c[:,11:12]
        self.m11_c = X_c[:,12:13]
        self.m12_c = X_c[:,13:14]
        self.m13_c = X_c[:,14:15] 
        self.m14_c = X_c[:,15:16]   
        self.m15_c = X_c[:,16:17] 
        self.m16_c = X_c[:,17:18] 
        self.m17_c = X_c[:,18:19] 
        self.m18_c = X_c[:,19:20] 
        self.m19_c = X_c[:,20:21]
        self.m20_c = X_c[:,21:22]
        self.m21_c = X_c[:,22:23]
        self.m22_c = X_c[:,23:24]
        self.m23_c = X_c[:,24:25] 
        self.m24_c = X_c[:,25:26] 
        self.m25_c = X_c[:,26:27] 
        self.m26_c = X_c[:,27:28] 
        self.m27_c = X_c[:,28:29] 
        self.m28_c = X_c[:,29:30] 
        self.m29_c = X_c[:,30:31]
        self.m30_c = X_c[:,31:32]
        self.m31_c = X_c[:,32:33]
        self.m32_c = X_c[:,33:34]
        self.m33_c = X_c[:,34:35] 
        self.m34_c = X_c[:,35:36]  
        self.m35_c = X_c[:,36:37]
        self.m36_c = X_c[:,37:38]
        self.m37_c = X_c[:,38:39]
        self.m38_c = X_c[:,39:40]
        self.m39_c = X_c[:,40:41]
        self.m40_c = X_c[:,41:42]
        self.m41_c = X_c[:,42:43]
        self.m42_c = X_c[:,43:44]
        self.m43_c = X_c[:,44:45] 
        self.m44_c = X_c[:,45:46]
        self.m45_c = X_c[:,46:47]
        self.m46_c = X_c[:,47:48]
        self.m47_c = X_c[:,48:49]
        self.m48_c = X_c[:,49:50]
        self.m49_c = X_c[:,50:51]
        self.m50_c = X_c[:,51:52]
        
        self.y0 = in_train[:,0:1]
        self.p0 = in_train[:,1:2]
        self.qa0 = in_train[:,2:3]
        self.qb0 = in_train[:,3:4]

        self.y_en = en_train[:,0:1]
        self.p_en = en_train[:,1:2]
        self.qa_en = en_train[:,2:3]
        self.qb_en = en_train[:,3:4]

        self.y_lb = lb_train[:,0:1]
        self.y_rb = rb_train[:,0:1]
        self.p_lb = lb_train[:,1:2]
        self.p_rb = rb_train[:,1:2] 
        self.qa_lb = lb_train[:,2:3]
        self.qa_rb = rb_train[:,2:3]
        self.qb_lb = lb_train[:,3:4]
        self.qb_rb = rb_train[:,3:4]
      
        self.layers = layers
        self.coeff0 = coeff[0:1]
        self.coeff1 = coeff[1:2]
        self.coeff2 = coeff[2:3]
        self.coeff3 = coeff[3:4]
        self.coeff4 = coeff[4:5]


        #Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        #Assign tensorflow placeholders and session
        config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        
        self.z0_tf = tf.placeholder(tf.float32, shape=[None, self.z0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.m10_tf = tf.placeholder(tf.float32, shape=[None, self.m10.shape[1]])
        self.m20_tf = tf.placeholder(tf.float32, shape=[None, self.m20.shape[1]]) 
        self.m30_tf = tf.placeholder(tf.float32, shape=[None, self.m30.shape[1]])  
        self.m40_tf = tf.placeholder(tf.float32, shape=[None, self.m40.shape[1]])  
        self.m50_tf = tf.placeholder(tf.float32, shape=[None, self.m50.shape[1]])
        self.m60_tf = tf.placeholder(tf.float32, shape=[None, self.m60.shape[1]])
        self.m70_tf = tf.placeholder(tf.float32, shape=[None, self.m70.shape[1]])
        self.m80_tf = tf.placeholder(tf.float32, shape=[None, self.m80.shape[1]])
        self.m90_tf = tf.placeholder(tf.float32, shape=[None, self.m90.shape[1]])
        self.m100_tf = tf.placeholder(tf.float32, shape=[None, self.m100.shape[1]])
        self.m110_tf = tf.placeholder(tf.float32, shape=[None, self.m110.shape[1]])
        self.m120_tf = tf.placeholder(tf.float32, shape=[None, self.m120.shape[1]]) 
        self.m130_tf = tf.placeholder(tf.float32, shape=[None, self.m130.shape[1]])  
        self.m140_tf = tf.placeholder(tf.float32, shape=[None, self.m140.shape[1]])  
        self.m150_tf = tf.placeholder(tf.float32, shape=[None, self.m150.shape[1]])
        self.m160_tf = tf.placeholder(tf.float32, shape=[None, self.m160.shape[1]])
        self.m170_tf = tf.placeholder(tf.float32, shape=[None, self.m170.shape[1]])
        self.m180_tf = tf.placeholder(tf.float32, shape=[None, self.m180.shape[1]])
        self.m190_tf = tf.placeholder(tf.float32, shape=[None, self.m190.shape[1]])
        self.m200_tf = tf.placeholder(tf.float32, shape=[None, self.m200.shape[1]])
        self.m210_tf = tf.placeholder(tf.float32, shape=[None, self.m210.shape[1]])
        self.m220_tf = tf.placeholder(tf.float32, shape=[None, self.m220.shape[1]]) 
        self.m230_tf = tf.placeholder(tf.float32, shape=[None, self.m230.shape[1]])  
        self.m240_tf = tf.placeholder(tf.float32, shape=[None, self.m240.shape[1]])  
        self.m250_tf = tf.placeholder(tf.float32, shape=[None, self.m250.shape[1]])
        self.m260_tf = tf.placeholder(tf.float32, shape=[None, self.m260.shape[1]])
        self.m270_tf = tf.placeholder(tf.float32, shape=[None, self.m270.shape[1]])
        self.m280_tf = tf.placeholder(tf.float32, shape=[None, self.m280.shape[1]])
        self.m290_tf = tf.placeholder(tf.float32, shape=[None, self.m290.shape[1]])
        self.m300_tf = tf.placeholder(tf.float32, shape=[None, self.m300.shape[1]])
        self.m310_tf = tf.placeholder(tf.float32, shape=[None, self.m310.shape[1]])
        self.m320_tf = tf.placeholder(tf.float32, shape=[None, self.m320.shape[1]]) 
        self.m330_tf = tf.placeholder(tf.float32, shape=[None, self.m330.shape[1]])  
        self.m340_tf = tf.placeholder(tf.float32, shape=[None, self.m340.shape[1]])  
        self.m350_tf = tf.placeholder(tf.float32, shape=[None, self.m350.shape[1]])
        self.m360_tf = tf.placeholder(tf.float32, shape=[None, self.m360.shape[1]])
        self.m370_tf = tf.placeholder(tf.float32, shape=[None, self.m370.shape[1]])
        self.m380_tf = tf.placeholder(tf.float32, shape=[None, self.m380.shape[1]])
        self.m390_tf = tf.placeholder(tf.float32, shape=[None, self.m390.shape[1]])
        self.m400_tf = tf.placeholder(tf.float32, shape=[None, self.m400.shape[1]])
        self.m410_tf = tf.placeholder(tf.float32, shape=[None, self.m410.shape[1]])
        self.m420_tf = tf.placeholder(tf.float32, shape=[None, self.m420.shape[1]]) 
        self.m430_tf = tf.placeholder(tf.float32, shape=[None, self.m430.shape[1]])  
        self.m440_tf = tf.placeholder(tf.float32, shape=[None, self.m440.shape[1]])  
        self.m450_tf = tf.placeholder(tf.float32, shape=[None, self.m450.shape[1]])
        self.m460_tf = tf.placeholder(tf.float32, shape=[None, self.m460.shape[1]])
        self.m470_tf = tf.placeholder(tf.float32, shape=[None, self.m470.shape[1]])
        self.m480_tf = tf.placeholder(tf.float32, shape=[None, self.m480.shape[1]])
        self.m490_tf = tf.placeholder(tf.float32, shape=[None, self.m490.shape[1]])
        self.m500_tf = tf.placeholder(tf.float32, shape=[None, self.m500.shape[1]])

        self.z_en_tf = tf.placeholder(tf.float32, shape=[None, self.z_en.shape[1]])
        self.t_en_tf = tf.placeholder(tf.float32, shape=[None, self.t_en.shape[1]])
        self.m1_en_tf = tf.placeholder(tf.float32, shape=[None, self.m1_en.shape[1]])
        self.m2_en_tf = tf.placeholder(tf.float32, shape=[None, self.m2_en.shape[1]]) 
        self.m3_en_tf = tf.placeholder(tf.float32, shape=[None, self.m3_en.shape[1]])
        self.m4_en_tf = tf.placeholder(tf.float32, shape=[None, self.m4_en.shape[1]])
        self.m5_en_tf = tf.placeholder(tf.float32, shape=[None, self.m5_en.shape[1]])
        self.m6_en_tf = tf.placeholder(tf.float32, shape=[None, self.m6_en.shape[1]])
        self.m7_en_tf = tf.placeholder(tf.float32, shape=[None, self.m7_en.shape[1]])
        self.m8_en_tf = tf.placeholder(tf.float32, shape=[None, self.m8_en.shape[1]])
        self.m9_en_tf = tf.placeholder(tf.float32, shape=[None, self.m9_en.shape[1]])
        self.m10_en_tf = tf.placeholder(tf.float32, shape=[None, self.m10_en.shape[1]])
        self.m11_en_tf = tf.placeholder(tf.float32, shape=[None, self.m11_en.shape[1]])
        self.m12_en_tf = tf.placeholder(tf.float32, shape=[None, self.m12_en.shape[1]]) 
        self.m13_en_tf = tf.placeholder(tf.float32, shape=[None, self.m13_en.shape[1]])  
        self.m14_en_tf = tf.placeholder(tf.float32, shape=[None, self.m14_en.shape[1]])  
        self.m15_en_tf = tf.placeholder(tf.float32, shape=[None, self.m15_en.shape[1]])
        self.m16_en_tf = tf.placeholder(tf.float32, shape=[None, self.m16_en.shape[1]])
        self.m17_en_tf = tf.placeholder(tf.float32, shape=[None, self.m17_en.shape[1]])
        self.m18_en_tf = tf.placeholder(tf.float32, shape=[None, self.m18_en.shape[1]])
        self.m19_en_tf = tf.placeholder(tf.float32, shape=[None, self.m19_en.shape[1]])
        self.m20_en_tf = tf.placeholder(tf.float32, shape=[None, self.m20_en.shape[1]])
        self.m21_en_tf = tf.placeholder(tf.float32, shape=[None, self.m21_en.shape[1]])
        self.m22_en_tf = tf.placeholder(tf.float32, shape=[None, self.m22_en.shape[1]]) 
        self.m23_en_tf = tf.placeholder(tf.float32, shape=[None, self.m23_en.shape[1]])  
        self.m24_en_tf = tf.placeholder(tf.float32, shape=[None, self.m24_en.shape[1]])  
        self.m25_en_tf = tf.placeholder(tf.float32, shape=[None, self.m25_en.shape[1]])
        self.m26_en_tf = tf.placeholder(tf.float32, shape=[None, self.m26_en.shape[1]])
        self.m27_en_tf = tf.placeholder(tf.float32, shape=[None, self.m27_en.shape[1]])
        self.m28_en_tf = tf.placeholder(tf.float32, shape=[None, self.m28_en.shape[1]])
        self.m29_en_tf = tf.placeholder(tf.float32, shape=[None, self.m29_en.shape[1]])
        self.m30_en_tf = tf.placeholder(tf.float32, shape=[None, self.m30_en.shape[1]])
        self.m31_en_tf = tf.placeholder(tf.float32, shape=[None, self.m31_en.shape[1]])
        self.m32_en_tf = tf.placeholder(tf.float32, shape=[None, self.m32_en.shape[1]]) 
        self.m33_en_tf = tf.placeholder(tf.float32, shape=[None, self.m33_en.shape[1]])  
        self.m34_en_tf = tf.placeholder(tf.float32, shape=[None, self.m34_en.shape[1]])  
        self.m35_en_tf = tf.placeholder(tf.float32, shape=[None, self.m35_en.shape[1]])
        self.m36_en_tf = tf.placeholder(tf.float32, shape=[None, self.m36_en.shape[1]])
        self.m37_en_tf = tf.placeholder(tf.float32, shape=[None, self.m37_en.shape[1]])
        self.m38_en_tf = tf.placeholder(tf.float32, shape=[None, self.m38_en.shape[1]])
        self.m39_en_tf = tf.placeholder(tf.float32, shape=[None, self.m39_en.shape[1]])
        self.m40_en_tf = tf.placeholder(tf.float32, shape=[None, self.m40_en.shape[1]])
        self.m41_en_tf = tf.placeholder(tf.float32, shape=[None, self.m41_en.shape[1]])
        self.m42_en_tf = tf.placeholder(tf.float32, shape=[None, self.m42_en.shape[1]]) 
        self.m43_en_tf = tf.placeholder(tf.float32, shape=[None, self.m43_en.shape[1]])  
        self.m44_en_tf = tf.placeholder(tf.float32, shape=[None, self.m44_en.shape[1]])  
        self.m45_en_tf = tf.placeholder(tf.float32, shape=[None, self.m45_en.shape[1]])
        self.m46_en_tf = tf.placeholder(tf.float32, shape=[None, self.m46_en.shape[1]])
        self.m47_en_tf = tf.placeholder(tf.float32, shape=[None, self.m47_en.shape[1]])
        self.m48_en_tf = tf.placeholder(tf.float32, shape=[None, self.m48_en.shape[1]])
        self.m49_en_tf = tf.placeholder(tf.float32, shape=[None, self.m49_en.shape[1]])
        self.m50_en_tf = tf.placeholder(tf.float32, shape=[None, self.m50_en.shape[1]])

        self.z_lb_tf = tf.placeholder(tf.float32, shape=[None, self.z_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]]) 
        self.m1_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m1_lb.shape[1]]) 
        self.m2_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m2_lb.shape[1]])
        self.m3_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m3_lb.shape[1]])
        self.m4_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m4_lb.shape[1]])
        self.m5_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m5_lb.shape[1]])
        self.m6_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m6_lb.shape[1]])
        self.m7_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m7_lb.shape[1]])
        self.m8_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m8_lb.shape[1]])
        self.m9_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m9_lb.shape[1]])
        self.m10_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m10_lb.shape[1]])
        self.m11_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m11_lb.shape[1]])
        self.m12_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m12_lb.shape[1]]) 
        self.m13_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m13_lb.shape[1]])  
        self.m14_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m14_lb.shape[1]])  
        self.m15_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m15_lb.shape[1]])
        self.m16_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m16_lb.shape[1]])
        self.m17_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m17_lb.shape[1]])
        self.m18_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m18_lb.shape[1]])
        self.m19_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m19_lb.shape[1]])
        self.m20_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m20_lb.shape[1]])
        self.m21_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m21_lb.shape[1]])
        self.m22_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m22_lb.shape[1]]) 
        self.m23_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m23_lb.shape[1]])  
        self.m24_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m24_lb.shape[1]])  
        self.m25_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m25_lb.shape[1]])
        self.m26_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m26_lb.shape[1]])
        self.m27_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m27_lb.shape[1]])
        self.m28_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m28_lb.shape[1]])
        self.m29_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m29_lb.shape[1]])
        self.m30_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m30_lb.shape[1]])
        self.m31_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m31_lb.shape[1]])
        self.m32_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m32_lb.shape[1]]) 
        self.m33_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m33_lb.shape[1]])  
        self.m34_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m34_lb.shape[1]])  
        self.m35_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m35_lb.shape[1]])
        self.m36_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m36_lb.shape[1]])
        self.m37_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m37_lb.shape[1]])
        self.m38_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m38_lb.shape[1]])
        self.m39_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m39_lb.shape[1]])
        self.m40_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m40_lb.shape[1]])
        self.m41_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m41_lb.shape[1]])
        self.m42_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m42_lb.shape[1]]) 
        self.m43_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m43_lb.shape[1]])  
        self.m44_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m44_lb.shape[1]])  
        self.m45_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m45_lb.shape[1]])
        self.m46_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m46_lb.shape[1]])
        self.m47_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m47_lb.shape[1]])
        self.m48_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m48_lb.shape[1]])
        self.m49_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m49_lb.shape[1]])
        self.m50_lb_tf = tf.placeholder(tf.float32, shape=[None, self.m50_lb.shape[1]])

        self.z_rb_tf = tf.placeholder(tf.float32, shape=[None, self.z_rb.shape[1]])
        self.t_rb_tf = tf.placeholder(tf.float32, shape=[None, self.t_rb.shape[1]]) 
        self.m1_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m1_rb.shape[1]]) 
        self.m2_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m2_rb.shape[1]])
        self.m3_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m3_rb.shape[1]])
        self.m4_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m4_rb.shape[1]])
        self.m5_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m5_rb.shape[1]])
        self.m6_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m6_rb.shape[1]])
        self.m7_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m7_rb.shape[1]])
        self.m8_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m8_rb.shape[1]])
        self.m9_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m9_rb.shape[1]])
        self.m10_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m10_rb.shape[1]])
        self.m11_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m11_rb.shape[1]])
        self.m12_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m12_rb.shape[1]]) 
        self.m13_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m13_rb.shape[1]])  
        self.m14_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m14_rb.shape[1]])  
        self.m15_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m15_rb.shape[1]])
        self.m16_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m16_rb.shape[1]])
        self.m17_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m17_rb.shape[1]])
        self.m18_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m18_rb.shape[1]])
        self.m19_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m19_rb.shape[1]])
        self.m20_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m20_rb.shape[1]])
        self.m21_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m21_rb.shape[1]])
        self.m22_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m22_rb.shape[1]]) 
        self.m23_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m23_rb.shape[1]])  
        self.m24_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m24_rb.shape[1]])  
        self.m25_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m25_rb.shape[1]])
        self.m26_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m26_rb.shape[1]])
        self.m27_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m27_rb.shape[1]])
        self.m28_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m28_rb.shape[1]])
        self.m29_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m29_rb.shape[1]])
        self.m30_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m30_rb.shape[1]])
        self.m31_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m31_rb.shape[1]])
        self.m32_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m32_rb.shape[1]]) 
        self.m33_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m33_rb.shape[1]])  
        self.m34_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m34_rb.shape[1]])  
        self.m35_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m35_rb.shape[1]])
        self.m36_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m36_rb.shape[1]])
        self.m37_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m37_rb.shape[1]])
        self.m38_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m38_rb.shape[1]])
        self.m39_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m39_rb.shape[1]])
        self.m40_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m40_rb.shape[1]])
        self.m41_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m41_rb.shape[1]])
        self.m42_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m42_rb.shape[1]]) 
        self.m43_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m43_rb.shape[1]])  
        self.m44_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m44_rb.shape[1]])  
        self.m45_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m45_rb.shape[1]])
        self.m46_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m46_rb.shape[1]])
        self.m47_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m47_rb.shape[1]])
        self.m48_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m48_rb.shape[1]])
        self.m49_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m49_rb.shape[1]])
        self.m50_rb_tf = tf.placeholder(tf.float32, shape=[None, self.m50_rb.shape[1]])

        self.y0_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.p0_tf = tf.placeholder(tf.float32, shape=[None, self.p0.shape[1]])
        self.qa0_tf = tf.placeholder(tf.float32, shape=[None, self.qa0.shape[1]])
        self.qb0_tf = tf.placeholder(tf.float32, shape=[None, self.qb0.shape[1]])

        self.y_en_tf = tf.placeholder(tf.float32, shape=[None, self.y_en.shape[1]])
        self.p_en_tf = tf.placeholder(tf.float32, shape=[None, self.p_en.shape[1]])
        self.qa_en_tf = tf.placeholder(tf.float32, shape=[None, self.qa_en.shape[1]])
        self.qb_en_tf = tf.placeholder(tf.float32, shape=[None, self.qb_en.shape[1]])

        self.y_lb_tf = tf.placeholder(tf.float32, shape=[None, self.y_lb.shape[1]])
        self.y_rb_tf = tf.placeholder(tf.float32, shape=[None, self.y_rb.shape[1]])
        self.p_lb_tf = tf.placeholder(tf.float32, shape=[None, self.p_lb.shape[1]])
        self.p_rb_tf = tf.placeholder(tf.float32, shape=[None, self.p_rb.shape[1]]) 
        self.qa_lb_tf = tf.placeholder(tf.float32, shape=[None, self.qa_lb.shape[1]])
        self.qa_rb_tf = tf.placeholder(tf.float32, shape=[None, self.qa_rb.shape[1]])
        self.qb_lb_tf = tf.placeholder(tf.float32, shape=[None, self.qb_lb.shape[1]])   
        self.qb_rb_tf = tf.placeholder(tf.float32, shape=[None, self.qb_rb.shape[1]])

        self.z_c_tf = tf.placeholder(tf.float32, shape=[None, self.z_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])   
        self.m1_c_tf = tf.placeholder(tf.float32, shape=[None, self.m1_c.shape[1]]) 
        self.m2_c_tf = tf.placeholder(tf.float32, shape=[None, self.m2_c.shape[1]]) 
        self.m3_c_tf = tf.placeholder(tf.float32, shape=[None, self.m3_c.shape[1]])  
        self.m4_c_tf = tf.placeholder(tf.float32, shape=[None, self.m4_c.shape[1]])
        self.m5_c_tf = tf.placeholder(tf.float32, shape=[None, self.m5_c.shape[1]])
        self.m6_c_tf = tf.placeholder(tf.float32, shape=[None, self.m6_c.shape[1]])
        self.m7_c_tf = tf.placeholder(tf.float32, shape=[None, self.m7_c.shape[1]])
        self.m8_c_tf = tf.placeholder(tf.float32, shape=[None, self.m8_c.shape[1]])
        self.m9_c_tf = tf.placeholder(tf.float32, shape=[None, self.m9_c.shape[1]])
        self.m10_c_tf = tf.placeholder(tf.float32, shape=[None, self.m10_c.shape[1]])
        self.m11_c_tf = tf.placeholder(tf.float32, shape=[None, self.m11_c.shape[1]])
        self.m12_c_tf = tf.placeholder(tf.float32, shape=[None, self.m12_c.shape[1]]) 
        self.m13_c_tf = tf.placeholder(tf.float32, shape=[None, self.m13_c.shape[1]])  
        self.m14_c_tf = tf.placeholder(tf.float32, shape=[None, self.m14_c.shape[1]])  
        self.m15_c_tf = tf.placeholder(tf.float32, shape=[None, self.m15_c.shape[1]])
        self.m16_c_tf = tf.placeholder(tf.float32, shape=[None, self.m16_c.shape[1]])
        self.m17_c_tf = tf.placeholder(tf.float32, shape=[None, self.m17_c.shape[1]])
        self.m18_c_tf = tf.placeholder(tf.float32, shape=[None, self.m18_c.shape[1]])
        self.m19_c_tf = tf.placeholder(tf.float32, shape=[None, self.m19_c.shape[1]])
        self.m20_c_tf = tf.placeholder(tf.float32, shape=[None, self.m20_c.shape[1]])
        self.m21_c_tf = tf.placeholder(tf.float32, shape=[None, self.m21_c.shape[1]])
        self.m22_c_tf = tf.placeholder(tf.float32, shape=[None, self.m22_c.shape[1]]) 
        self.m23_c_tf = tf.placeholder(tf.float32, shape=[None, self.m23_c.shape[1]])  
        self.m24_c_tf = tf.placeholder(tf.float32, shape=[None, self.m24_c.shape[1]])  
        self.m25_c_tf = tf.placeholder(tf.float32, shape=[None, self.m25_c.shape[1]])
        self.m26_c_tf = tf.placeholder(tf.float32, shape=[None, self.m26_c.shape[1]])
        self.m27_c_tf = tf.placeholder(tf.float32, shape=[None, self.m27_c.shape[1]])
        self.m28_c_tf = tf.placeholder(tf.float32, shape=[None, self.m28_c.shape[1]])
        self.m29_c_tf = tf.placeholder(tf.float32, shape=[None, self.m29_c.shape[1]])
        self.m30_c_tf = tf.placeholder(tf.float32, shape=[None, self.m30_c.shape[1]])
        self.m31_c_tf = tf.placeholder(tf.float32, shape=[None, self.m31_c.shape[1]])
        self.m32_c_tf = tf.placeholder(tf.float32, shape=[None, self.m32_c.shape[1]]) 
        self.m33_c_tf = tf.placeholder(tf.float32, shape=[None, self.m33_c.shape[1]])  
        self.m34_c_tf = tf.placeholder(tf.float32, shape=[None, self.m34_c.shape[1]])  
        self.m35_c_tf = tf.placeholder(tf.float32, shape=[None, self.m35_c.shape[1]])
        self.m36_c_tf = tf.placeholder(tf.float32, shape=[None, self.m36_c.shape[1]])
        self.m37_c_tf = tf.placeholder(tf.float32, shape=[None, self.m37_c.shape[1]])
        self.m38_c_tf = tf.placeholder(tf.float32, shape=[None, self.m38_c.shape[1]])
        self.m39_c_tf = tf.placeholder(tf.float32, shape=[None, self.m39_c.shape[1]])
        self.m40_c_tf = tf.placeholder(tf.float32, shape=[None, self.m40_c.shape[1]])
        self.m41_c_tf = tf.placeholder(tf.float32, shape=[None, self.m41_c.shape[1]])
        self.m42_c_tf = tf.placeholder(tf.float32, shape=[None, self.m42_c.shape[1]]) 
        self.m43_c_tf = tf.placeholder(tf.float32, shape=[None, self.m43_c.shape[1]])  
        self.m44_c_tf = tf.placeholder(tf.float32, shape=[None, self.m44_c.shape[1]])  
        self.m45_c_tf = tf.placeholder(tf.float32, shape=[None, self.m45_c.shape[1]])
        self.m46_c_tf = tf.placeholder(tf.float32, shape=[None, self.m46_c.shape[1]])
        self.m47_c_tf = tf.placeholder(tf.float32, shape=[None, self.m47_c.shape[1]])
        self.m48_c_tf = tf.placeholder(tf.float32, shape=[None, self.m48_c.shape[1]])
        self.m49_c_tf = tf.placeholder(tf.float32, shape=[None, self.m49_c.shape[1]])
        self.m50_c_tf = tf.placeholder(tf.float32, shape=[None, self.m50_c.shape[1]])

        self.N0 = tf.cast(self.N0, dtype=tf.int32)
        self.N_b = tf.cast(self.N_b, dtype=tf.int32)
        self.N_c = tf.cast(self.N_c, dtype=tf.int32)    
        
        self.y0_pred, self.p0_pred, self.qa0_pred, self.qb0_pred = self.net_ads(self.z0_tf, self.t0_tf, self.m10_tf, self.m20_tf, self.m30_tf, self.m40_tf, 
                                                                                self.m50_tf, self.m60_tf, self.m70_tf, self.m80_tf, self.m90_tf, self.m100_tf,
                                                                                self.m110_tf, self.m120_tf, self.m130_tf, self.m140_tf, self.m150_tf, 
                                                                                self.m160_tf, self.m170_tf, self.m180_tf, self.m190_tf, self.m200_tf,
                                                                                self.m210_tf, self.m220_tf, self.m230_tf, self.m240_tf, self.m250_tf, 
                                                                                self.m260_tf, self.m270_tf, self.m280_tf, self.m290_tf, self.m300_tf,
                                                                                self.m310_tf, self.m320_tf, self.m330_tf, self.m340_tf, self.m350_tf, 
                                                                                self.m360_tf, self.m370_tf, self.m380_tf, self.m390_tf, self.m400_tf,
                                                                                self.m410_tf, self.m420_tf, self.m430_tf, self.m440_tf, self.m450_tf, 
                                                                                self.m460_tf, self.m470_tf, self.m480_tf, self.m490_tf, self.m500_tf) 
        self.y_en_pred, self.p_en_pred, self.qa_en_pred, self.qb_en_pred = self.net_ads(self.z_en_tf, self.t_en_tf, self.m1_en_tf, self.m2_en_tf, 
                                                                                        self.m3_en_tf, self.m4_en_tf, self.m5_en_tf, self.m6_en_tf, 
                                                                                        self.m7_en_tf, self.m8_en_tf, self.m9_en_tf, self.m10_en_tf, 
                                                                                        self.m11_en_tf, self.m12_en_tf, self.m13_en_tf, self.m14_en_tf, 
                                                                                        self.m15_en_tf, self.m16_en_tf, self.m17_en_tf, self.m18_en_tf, 
                                                                                        self.m19_en_tf, self.m20_en_tf, self.m21_en_tf, self.m22_en_tf, 
                                                                                        self.m23_en_tf, self.m24_en_tf, self.m25_en_tf, self.m26_en_tf, 
                                                                                        self.m27_en_tf, self.m28_en_tf, self.m29_en_tf, self.m30_en_tf,
                                                                                        self.m31_en_tf, self.m32_en_tf, self.m33_en_tf, self.m34_en_tf, 
                                                                                        self.m35_en_tf, self.m36_en_tf, self.m37_en_tf, self.m38_en_tf, 
                                                                                        self.m39_en_tf, self.m40_en_tf, self.m41_en_tf, self.m42_en_tf, 
                                                                                        self.m43_en_tf, self.m44_en_tf, self.m45_en_tf, self.m46_en_tf, 
                                                                                        self.m47_en_tf, self.m48_en_tf, self.m49_en_tf, self.m50_en_tf) 
        self.y_lb_pred, self.p_lb_pred, self.qa_lb_pred, self.qb_lb_pred = self.net_ads(self.z_lb_tf, self.t_lb_tf, self.m1_lb_tf, self.m2_lb_tf, 
                                                                                        self.m3_lb_tf, self.m4_lb_tf, self.m5_lb_tf, self.m6_lb_tf, 
                                                                                        self.m7_lb_tf, self.m8_lb_tf, self.m9_lb_tf, self.m10_lb_tf,
                                                                                        self.m11_lb_tf, self.m12_lb_tf, self.m13_lb_tf, self.m14_lb_tf, 
                                                                                        self.m15_lb_tf, self.m16_lb_tf, self.m17_lb_tf, self.m18_lb_tf, 
                                                                                        self.m19_lb_tf, self.m20_lb_tf, self.m21_lb_tf, self.m22_lb_tf, 
                                                                                        self.m23_lb_tf, self.m24_lb_tf, self.m25_lb_tf, self.m26_lb_tf, 
                                                                                        self.m27_lb_tf, self.m28_lb_tf, self.m29_lb_tf, self.m30_lb_tf, 
                                                                                        self.m31_lb_tf, self.m32_lb_tf, self.m33_lb_tf, self.m34_lb_tf, 
                                                                                        self.m35_lb_tf, self.m36_lb_tf, self.m37_lb_tf, self.m38_lb_tf, 
                                                                                        self.m39_lb_tf, self.m40_lb_tf, self.m41_lb_tf, self.m42_lb_tf, 
                                                                                        self.m43_lb_tf, self.m44_lb_tf, self.m45_lb_tf, self.m46_lb_tf, 
                                                                                        self.m47_lb_tf, self.m48_lb_tf, self.m49_lb_tf, self.m50_lb_tf) 
        self.y_rb_pred, self.p_rb_pred, self.qa_rb_pred, self.qb_rb_pred = self.net_ads(self.z_rb_tf, self.t_rb_tf, self.m1_rb_tf, self.m2_rb_tf, 
                                                                                        self.m3_rb_tf, self.m4_rb_tf, self.m5_rb_tf, self.m6_rb_tf, 
                                                                                        self.m7_rb_tf, self.m8_rb_tf, self.m9_rb_tf, self.m10_rb_tf, 
                                                                                        self.m11_rb_tf, self.m12_rb_tf, self.m13_rb_tf, self.m14_rb_tf, 
                                                                                        self.m15_rb_tf, self.m16_rb_tf, self.m17_rb_tf, self.m18_rb_tf, 
                                                                                        self.m19_rb_tf, self.m20_rb_tf, self.m21_rb_tf, self.m22_rb_tf, 
                                                                                        self.m23_rb_tf, self.m24_rb_tf, self.m25_rb_tf, self.m26_rb_tf, 
                                                                                        self.m27_rb_tf, self.m28_rb_tf, self.m29_rb_tf, self.m30_rb_tf,
                                                                                        self.m31_rb_tf, self.m32_rb_tf, self.m33_rb_tf, self.m34_rb_tf, 
                                                                                        self.m35_rb_tf, self.m36_rb_tf, self.m37_rb_tf, self.m38_rb_tf, 
                                                                                        self.m39_rb_tf, self.m40_rb_tf, self.m41_rb_tf, self.m42_rb_tf, 
                                                                                        self.m43_rb_tf, self.m44_rb_tf, self.m45_rb_tf, self.m46_rb_tf, 
                                                                                        self.m47_rb_tf, self.m48_rb_tf, self.m49_rb_tf, self.m50_rb_tf)
        self.f_y_pred, self.f_p_pred = self.net_f(self.z_c_tf, self.t_c_tf, self.m1_c_tf, self.m2_c_tf, self.m3_c_tf, self.m4_c_tf, 
                                                  self.m5_c_tf, self.m6_c_tf, self.m7_c_tf, self.m8_c_tf, self.m9_c_tf, self.m10_c_tf,
                                                  self.m11_c_tf, self.m12_c_tf, self.m13_c_tf, self.m14_c_tf, self.m15_c_tf, 
                                                  self.m16_c_tf, self.m17_c_tf, self.m18_c_tf, self.m19_c_tf, self.m20_c_tf,
                                                  self.m21_c_tf, self.m22_c_tf, self.m23_c_tf, self.m24_c_tf, self.m25_c_tf, 
                                                  self.m26_c_tf, self.m27_c_tf, self.m28_c_tf, self.m29_c_tf, self.m30_c_tf,
                                                  self.m31_c_tf, self.m32_c_tf, self.m33_c_tf, self.m34_c_tf, self.m35_c_tf, 
                                                  self.m36_c_tf, self.m37_c_tf, self.m38_c_tf, self.m39_c_tf, self.m40_c_tf,
                                                  self.m41_c_tf, self.m42_c_tf, self.m43_c_tf, self.m44_c_tf, self.m45_c_tf, 
                                                  self.m46_c_tf, self.m47_c_tf, self.m48_c_tf, self.m49_c_tf, self.m50_c_tf) 

        
        self.loss = 100*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.y0_tf - self.y0_pred), self.N0, self.N_k)) + \
                    100*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.p0_tf - self.p0_pred), self.N0, self.N_k)) + \
                    1*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.qa0_tf - self.qa0_pred), self.N0, self.N_k)) + \
                    1*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.qb0_tf - self.qb0_pred), self.N0, self.N_k)) + \
                    100*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.y_en_tf - self.y_en_pred), self.N0, self.N_k)) + \
                    100*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.y_lb_tf - self.y_lb_pred), self.N_b, self.N_k)) + \
                    100*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.p_lb_tf - self.p_lb_pred), self.N_b, self.N_k)) + \
                    1*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.qa_lb_tf - self.qa_lb_pred), self.N_b, self.N_k)) + \
                    1*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.qb_lb_tf - self.qb_lb_pred), self.N_b, self.N_k)) + \
                    100*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.y_rb_tf - self.y_rb_pred), self.N_b, self.N_k)) + \
                    100*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.p_rb_tf - self.p_rb_pred), self.N_b, self.N_k)) + \
                    1*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.qa_rb_tf - self.qa_rb_pred), self.N_b, self.N_k)) + \
                    1*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.qb_rb_tf - self.qb_rb_pred), self.N_b, self.N_k)) + \
                    0.04*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.f_y_pred), self.N_c, self.N_k)) + \
                    0.04*tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.f_p_pred), self.N_c, self.N_k)) 
      

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
 
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()


    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.random.normal([1,layers[l+1]], 0, 1, dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)  
        return weights, biases
      
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
      
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.low_bound)/(self.up_bound - self.low_bound) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.multiply(1.0,tf.add(tf.matmul(H, W), b))) #change to tanh for adsorption step
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b, name="Y_out")
        return Y
      
    def net_ads(self, z, t, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10,
                m11, m12, m13, m14, m15, m16, m17, m18, m19, m20,
                m21, m22, m23, m24, m25, m26, m27, m28, m29, m30,
                m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
                m41, m42, m43, m44, m45, m46, m47, m48, m49, m50):
        X = tf.concat([z,t,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,
                       m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
                       m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,
                       m31,m32,m33,m34,m35,m36,m37,m38,m39,m40,
                       m41,m42,m43,m44,m45,m46,m47,m48,m49,m50],1)
        ads = self.neural_net(X, self.weights, self.biases)
        y = ads[:,0:1]
        p = ads[:,1:2]
        qa = ads[:,2:3]
        qb = ads[:,3:4]
        
        return y, p, qa, qb

    def net_f(self, z, t, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10,
                m11, m12, m13, m14, m15, m16, m17, m18, m19, m20,
                m21, m22, m23, m24, m25, m26, m27, m28, m29, m30,
                m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
                m41, m42, m43, m44, m45, m46, m47, m48, m49, m50):
        y, p, qa, qb = self.net_ads(z,t,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,
                                    m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,
                                    m21,m22,m23,m24,m25,m26,m27,m28,m29,m30,
                                    m31,m32,m33,m34,m35,m36,m37,m38,m39,m40,
                                    m41,m42,m43,m44,m45,m46,m47,m48,m49,m50)
        
        c = y*p   

        c_t = tf.gradients(c, t)[0]
        p_t = tf.gradients(p, t)[0]
        c_z = tf.gradients(c, z)[0] 
        p_z = tf.gradients(p, z)[0] 
        qa_t = tf.gradients(qa, t)[0]
        qb_t = tf.gradients(qb, t)[0]
        c_zz = tf.gradients(c_z, z)[0] 
        
        ve = -self.coeff0*p_z
        jp_z = tf.gradients(p*ve, z)[0]
        jc_z = tf.gradients(c*ve, z)[0]
 
        f_y = c_t + self.coeff1*jc_z - self.coeff2*c_zz + self.coeff3*qa_t
        f_p = p_t + self.coeff1*jp_z + self.coeff3*qa_t + self.coeff4*qb_t        
        return f_y, f_p
      
      
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self):
        
        tf_dict = {self.z0_tf: self.z0, self.t0_tf: self.t0, self.m10_tf: self.m10, self.m20_tf: self.m20, self.m30_tf: self.m30, 
                   self.m40_tf: self.m40, self.m50_tf: self.m50, self.m60_tf: self.m60, self.m70_tf: self.m70,
                   self.m80_tf: self.m80, self.m90_tf: self.m90, self.m100_tf: self.m100, self.m110_tf: self.m110, self.m120_tf: self.m120, 
                   self.m130_tf: self.m130, self.m140_tf: self.m140, self.m150_tf: self.m150, self.m160_tf: self.m160, self.m170_tf: self.m170,
                   self.m180_tf: self.m180, self.m190_tf: self.m190, self.m200_tf: self.m200, self.m210_tf: self.m210, self.m220_tf: self.m220, 
                   self.m230_tf: self.m230, self.m240_tf: self.m240, self.m250_tf: self.m250, self.m260_tf: self.m260, self.m270_tf: self.m270,
                   self.m280_tf: self.m280, self.m290_tf: self.m290, self.m300_tf: self.m300, self.m310_tf: self.m310, self.m320_tf: self.m320, 
                   self.m330_tf: self.m330, self.m340_tf: self.m340, self.m350_tf: self.m350, self.m360_tf: self.m360, self.m370_tf: self.m370,
                   self.m380_tf: self.m380, self.m390_tf: self.m390, self.m400_tf: self.m400, self.m410_tf: self.m410, self.m420_tf: self.m420, 
                   self.m430_tf: self.m430, self.m440_tf: self.m440, self.m450_tf: self.m450, self.m460_tf: self.m460, self.m470_tf: self.m470,
                   self.m480_tf: self.m480, self.m490_tf: self.m490, self.m500_tf: self.m500, self.z_en_tf: self.z_en, self.t_en_tf: self.t_en,
                   self.m1_en_tf: self.m1_en, self.m2_en_tf: self.m2_en, self.m3_en_tf: self.m3_en, self.m4_en_tf: self.m4_en, 
                   self.m5_en_tf: self.m5_en, self.m6_en_tf: self.m6_en, self.m7_en_tf: self.m7_en, self.m8_en_tf: self.m8_en, 
                   self.m9_en_tf: self.m9_en, self.m10_en_tf: self.m10_en, self.m11_en_tf: self.m11_en, self.m12_en_tf: self.m12_en, 
                   self.m13_en_tf: self.m13_en, self.m14_en_tf: self.m14_en, self.m15_en_tf: self.m15_en, self.m16_en_tf: self.m16_en, 
                   self.m17_en_tf: self.m17_en, self.m18_en_tf: self.m18_en, self.m19_en_tf: self.m19_en, self.m20_en_tf: self.m20_en, 
                   self.m21_en_tf: self.m21_en, self.m22_en_tf: self.m22_en, self.m23_en_tf: self.m23_en, self.m24_en_tf: self.m24_en, 
                   self.m25_en_tf: self.m25_en, self.m26_en_tf: self.m26_en, self.m27_en_tf: self.m27_en, self.m28_en_tf: self.m28_en, 
                   self.m29_en_tf: self.m29_en, self.m30_en_tf: self.m30_en, self.m31_en_tf: self.m31_en, self.m32_en_tf: self.m32_en,
                   self.m33_en_tf: self.m33_en, self.m34_en_tf: self.m34_en, self.m35_en_tf: self.m35_en, self.m36_en_tf: self.m36_en,
                   self.m37_en_tf: self.m37_en, self.m38_en_tf: self.m38_en, self.m39_en_tf: self.m39_en, self.m40_en_tf: self.m40_en, 
                   self.m41_en_tf: self.m41_en, self.m42_en_tf: self.m42_en, self.m43_en_tf: self.m43_en, self.m44_en_tf: self.m44_en,
                   self.m45_en_tf: self.m45_en, self.m46_en_tf: self.m46_en, self.m47_en_tf: self.m47_en, self.m48_en_tf: self.m48_en,
                   self.m49_en_tf: self.m49_en, self.m50_en_tf: self.m50_en, self.z_lb_tf: self.z_lb, self.t_lb_tf: self.t_lb, 
                   self.m1_lb_tf: self.m1_lb, self.m2_lb_tf: self.m2_lb, self.m3_lb_tf: self.m3_lb, self.m4_lb_tf: self.m4_lb, 
                   self.m5_lb_tf: self.m5_lb, self.m6_lb_tf: self.m6_lb, self.m7_lb_tf: self.m7_lb, self.m8_lb_tf: self.m8_lb, 
                   self.m9_lb_tf: self.m9_lb, self.m10_lb_tf: self.m10_lb, self.m11_lb_tf: self.m11_lb, self.m12_lb_tf: self.m12_lb,
                   self.m13_lb_tf: self.m13_lb, self.m14_lb_tf: self.m14_lb, self.m15_lb_tf: self.m15_lb, self.m16_lb_tf: self.m16_lb,
                   self.m17_lb_tf: self.m17_lb, self.m18_lb_tf: self.m18_lb, self.m19_lb_tf: self.m19_lb, self.m20_lb_tf: self.m20_lb,
                   self.m21_lb_tf: self.m21_lb, self.m22_lb_tf: self.m22_lb, self.m23_lb_tf: self.m23_lb, self.m24_lb_tf: self.m24_lb,
                   self.m25_lb_tf: self.m25_lb, self.m26_lb_tf: self.m26_lb, self.m27_lb_tf: self.m27_lb, self.m28_lb_tf: self.m28_lb,
                   self.m29_lb_tf: self.m29_lb, self.m30_lb_tf: self.m30_lb, self.m31_lb_tf: self.m31_lb, self.m32_lb_tf: self.m32_lb,
                   self.m33_lb_tf: self.m33_lb, self.m34_lb_tf: self.m34_lb, self.m35_lb_tf: self.m35_lb, self.m36_lb_tf: self.m36_lb,
                   self.m37_lb_tf: self.m37_lb, self.m38_lb_tf: self.m38_lb, self.m39_lb_tf: self.m39_lb, self.m40_lb_tf: self.m40_lb,
                   self.m41_lb_tf: self.m41_lb, self.m42_lb_tf: self.m42_lb, self.m43_lb_tf: self.m43_lb, self.m44_lb_tf: self.m44_lb,
                   self.m45_lb_tf: self.m45_lb, self.m46_lb_tf: self.m46_lb, self.m47_lb_tf: self.m47_lb, self.m48_lb_tf: self.m48_lb,
                   self.m49_lb_tf: self.m49_lb, self.m50_lb_tf: self.m50_lb, self.z_rb_tf: self.z_rb, self.t_rb_tf: self.t_rb, 
                   self.m1_rb_tf: self.m1_rb, self.m2_rb_tf: self.m2_rb, self.m3_rb_tf: self.m3_rb, self.m4_rb_tf: self.m4_rb, 
                   self.m5_rb_tf: self.m5_rb, self.m6_rb_tf: self.m6_rb, self.m7_rb_tf: self.m7_rb, self.m8_rb_tf: self.m8_rb,
                   self.m9_rb_tf: self.m9_rb, self.m10_rb_tf: self.m10_rb, self.m11_rb_tf: self.m11_rb, self.m12_rb_tf: self.m12_rb,
                   self.m13_rb_tf: self.m13_rb, self.m14_rb_tf: self.m14_rb, self.m15_rb_tf: self.m15_rb, self.m16_rb_tf: self.m16_rb,
                   self.m17_rb_tf: self.m17_rb, self.m18_rb_tf: self.m18_rb, self.m19_rb_tf: self.m19_rb, self.m20_rb_tf: self.m20_rb,
                   self.m21_rb_tf: self.m21_rb, self.m22_rb_tf: self.m22_rb, self.m23_rb_tf: self.m23_rb, self.m24_rb_tf: self.m24_rb, 
                   self.m25_rb_tf: self.m25_rb, self.m26_rb_tf: self.m26_rb, self.m27_rb_tf: self.m27_rb, self.m28_rb_tf: self.m28_rb,
                   self.m29_rb_tf: self.m29_rb, self.m30_rb_tf: self.m30_rb, self.m31_rb_tf: self.m31_rb, self.m32_rb_tf: self.m32_rb,
                   self.m33_rb_tf: self.m33_rb, self.m34_rb_tf: self.m34_rb, self.m35_rb_tf: self.m35_rb, self.m36_rb_tf: self.m36_rb,
                   self.m37_rb_tf: self.m37_rb, self.m38_rb_tf: self.m38_rb, self.m39_rb_tf: self.m39_rb, self.m40_rb_tf: self.m40_rb,
                   self.m41_rb_tf: self.m41_rb, self.m42_rb_tf: self.m42_rb, self.m43_rb_tf: self.m43_rb, self.m44_rb_tf: self.m44_rb,
                   self.m45_rb_tf: self.m45_rb, self.m46_rb_tf: self.m46_rb, self.m47_rb_tf: self.m47_rb, self.m48_rb_tf: self.m48_rb,
                   self.m49_rb_tf: self.m49_rb, self.m50_rb_tf: self.m50_rb, self.qa0_tf: self.qa0, self.qb0_tf: self.qb0, self.y0_tf: self.y0,
                   self.p0_tf: self.p0, self.qa_en_tf: self.qa_en, self.qb_en_tf: self.qb_en, self.y_en_tf: self.y_en, self.p_en_tf: self.p_en, 
                   self.qa_lb_tf: self.qa_lb, self.qb_lb_tf: self.qb_lb, self.y_lb_tf: self.y_lb, self.p_lb_tf: self.p_lb, 
                   self.qa_rb_tf: self.qa_rb, self.qb_rb_tf: self.qb_rb, self.y_rb_tf: self.y_rb, self.p_rb_tf: self.p_rb,      
                   self.z_c_tf: self.z_c, self.t_c_tf: self.t_c, self.m1_c_tf: self.m1_c, self.m2_c_tf: self.m2_c, self.m3_c_tf: self.m3_c, 
                   self.m4_c_tf: self.m4_c, self.m5_c_tf: self.m5_c, self.m6_c_tf: self.m6_c, self.m7_c_tf: self.m7_c, self.m8_c_tf: self.m8_c,
                   self.m9_c_tf: self.m9_c, self.m10_c_tf: self.m10_c, self.m11_c_tf: self.m11_c, self.m12_c_tf: self.m12_c,
                   self.m13_c_tf: self.m13_c, self.m14_c_tf: self.m14_c, self.m15_c_tf: self.m15_c, self.m16_c_tf: self.m16_c, 
                   self.m17_c_tf: self.m17_c, self.m18_c_tf: self.m18_c, self.m19_c_tf: self.m19_c, self.m20_c_tf: self.m20_c, 
                   self.m21_c_tf: self.m21_c, self.m22_c_tf: self.m22_c, self.m23_c_tf: self.m23_c, self.m24_c_tf: self.m24_c,
                   self.m25_c_tf: self.m25_c, self.m26_c_tf: self.m26_c, self.m27_c_tf: self.m27_c, self.m28_c_tf: self.m28_c,
                   self.m29_c_tf: self.m29_c, self.m30_c_tf: self.m30_c, self.m31_c_tf: self.m31_c, self.m32_c_tf: self.m32_c,
                   self.m33_c_tf: self.m33_c, self.m34_c_tf: self.m34_c, self.m35_c_tf: self.m35_c, self.m36_c_tf: self.m36_c,
                   self.m37_c_tf: self.m37_c, self.m38_c_tf: self.m38_c, self.m39_c_tf: self.m39_c, self.m40_c_tf: self.m40_c, 
                   self.m41_c_tf: self.m41_c, self.m42_c_tf: self.m42_c, self.m43_c_tf: self.m43_c, self.m44_c_tf: self.m44_c,
                   self.m45_c_tf: self.m45_c, self.m46_c_tf: self.m46_c, self.m47_c_tf: self.m47_c, self.m48_c_tf: self.m48_c,
                   self.m49_c_tf: self.m49_c, self.m50_c_tf: self.m50_c}
                   
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss],
                                loss_callback = self.callback) 

    def save(self):
        saved_model = self.saver.save(self.sess, 'model')
        return saved_model

    def predict(self, X_hat):
                
        y_hat = self.sess.run(self.y_lb_pred, {self.z_lb_tf: X_hat[:,0:1], self.t_lb_tf: X_hat[:,1:2], self.m1_lb_tf: X_hat[:,2:3], 
                                                self.m2_lb_tf: X_hat[:,3:4], self.m3_lb_tf: X_hat[:,4:5], self.m4_lb_tf: X_hat[:,5:6], 
                                                self.m5_lb_tf: X_hat[:,6:7], self.m6_lb_tf: X_hat[:,7:8], self.m7_lb_tf: X_hat[:,8:9],
                                                self.m8_lb_tf: X_hat[:,9:10], self.m9_lb_tf: X_hat[:,10:11], self.m10_lb_tf: X_hat[:,11:12],
                                                self.m11_lb_tf: X_hat[:,12:13], self.m12_lb_tf: X_hat[:,13:14], self.m13_lb_tf: X_hat[:,14:15], 
                                                self.m14_lb_tf: X_hat[:,15:16], self.m15_lb_tf: X_hat[:,16:17], self.m16_lb_tf: X_hat[:,17:18], 
                                                self.m17_lb_tf: X_hat[:,18:19], self.m18_lb_tf: X_hat[:,19:20], self.m19_lb_tf: X_hat[:,20:21], 
                                                self.m20_lb_tf: X_hat[:,21:22], self.m21_lb_tf: X_hat[:,22:23], self.m22_lb_tf: X_hat[:,23:24],
                                                self.m23_lb_tf: X_hat[:,24:25], self.m24_lb_tf: X_hat[:,25:26], self.m25_lb_tf: X_hat[:,26:27], 
                                                self.m26_lb_tf: X_hat[:,27:28], self.m27_lb_tf: X_hat[:,28:29], self.m28_lb_tf: X_hat[:,29:30], 
                                                self.m29_lb_tf: X_hat[:,30:31], self.m30_lb_tf: X_hat[:,31:32], self.m31_lb_tf: X_hat[:,32:33], 
                                                self.m32_lb_tf: X_hat[:,33:34], self.m33_lb_tf: X_hat[:,34:35], self.m34_lb_tf: X_hat[:,35:36],
                                                self.m35_lb_tf: X_hat[:,36:37], self.m36_lb_tf: X_hat[:,37:38], self.m37_lb_tf: X_hat[:,38:39],
                                                self.m38_lb_tf: X_hat[:,39:40], self.m39_lb_tf: X_hat[:,40:41], self.m40_lb_tf: X_hat[:,41:42],
                                                self.m41_lb_tf: X_hat[:,42:43], self.m42_lb_tf: X_hat[:,43:44], self.m43_lb_tf: X_hat[:,44:45], 
                                                self.m44_lb_tf: X_hat[:,45:46], self.m45_lb_tf: X_hat[:,46:47], self.m46_lb_tf: X_hat[:,47:48],
                                                self.m47_lb_tf: X_hat[:,48:49], self.m48_lb_tf: X_hat[:,49:50], self.m49_lb_tf: X_hat[:,50:51],
                                                self.m50_lb_tf: X_hat[:,51:52]}) 
        p_hat = self.sess.run(self.p_lb_pred, {self.z_lb_tf: X_hat[:,0:1], self.t_lb_tf: X_hat[:,1:2], self.m1_lb_tf: X_hat[:,2:3], 
                                                self.m2_lb_tf: X_hat[:,3:4], self.m3_lb_tf: X_hat[:,4:5], self.m4_lb_tf: X_hat[:,5:6], 
                                                self.m5_lb_tf: X_hat[:,6:7], self.m6_lb_tf: X_hat[:,7:8], self.m7_lb_tf: X_hat[:,8:9],
                                                self.m8_lb_tf: X_hat[:,9:10], self.m9_lb_tf: X_hat[:,10:11], self.m10_lb_tf: X_hat[:,11:12],
                                                self.m11_lb_tf: X_hat[:,12:13], self.m12_lb_tf: X_hat[:,13:14], self.m13_lb_tf: X_hat[:,14:15], 
                                                self.m14_lb_tf: X_hat[:,15:16], self.m15_lb_tf: X_hat[:,16:17], self.m16_lb_tf: X_hat[:,17:18], 
                                                self.m17_lb_tf: X_hat[:,18:19], self.m18_lb_tf: X_hat[:,19:20], self.m19_lb_tf: X_hat[:,20:21], 
                                                self.m20_lb_tf: X_hat[:,21:22], self.m21_lb_tf: X_hat[:,22:23], self.m22_lb_tf: X_hat[:,23:24],
                                                self.m23_lb_tf: X_hat[:,24:25], self.m24_lb_tf: X_hat[:,25:26], self.m25_lb_tf: X_hat[:,26:27], 
                                                self.m26_lb_tf: X_hat[:,27:28], self.m27_lb_tf: X_hat[:,28:29], self.m28_lb_tf: X_hat[:,29:30], 
                                                self.m29_lb_tf: X_hat[:,30:31], self.m30_lb_tf: X_hat[:,31:32], self.m31_lb_tf: X_hat[:,32:33], 
                                                self.m32_lb_tf: X_hat[:,33:34], self.m33_lb_tf: X_hat[:,34:35], self.m34_lb_tf: X_hat[:,35:36],
                                                self.m35_lb_tf: X_hat[:,36:37], self.m36_lb_tf: X_hat[:,37:38], self.m37_lb_tf: X_hat[:,38:39],
                                                self.m38_lb_tf: X_hat[:,39:40], self.m39_lb_tf: X_hat[:,40:41], self.m40_lb_tf: X_hat[:,41:42],
                                                self.m41_lb_tf: X_hat[:,42:43], self.m42_lb_tf: X_hat[:,43:44], self.m43_lb_tf: X_hat[:,44:45], 
                                                self.m44_lb_tf: X_hat[:,45:46], self.m45_lb_tf: X_hat[:,46:47], self.m46_lb_tf: X_hat[:,47:48],
                                                self.m47_lb_tf: X_hat[:,48:49], self.m48_lb_tf: X_hat[:,49:50], self.m49_lb_tf: X_hat[:,50:51],
                                                self.m50_lb_tf: X_hat[:,51:52]})                                           
        qa_hat = self.sess.run(self.qa0_pred, {self.z0_tf: X_hat[:,0:1], self.t0_tf: X_hat[:,1:2], self.m10_tf: X_hat[:,2:3], 
                                                self.m20_tf: X_hat[:,3:4], self.m30_tf: X_hat[:,4:5], self.m40_tf: X_hat[:,5:6], 
                                                self.m50_tf: X_hat[:,6:7], self.m60_tf: X_hat[:,7:8], self.m70_tf: X_hat[:,8:9],
                                                self.m80_tf: X_hat[:,9:10], self.m90_tf: X_hat[:,10:11], self.m100_tf: X_hat[:,11:12],
                                                self.m110_tf: X_hat[:,12:13], self.m120_tf: X_hat[:,13:14], self.m130_tf: X_hat[:,14:15], 
                                                self.m140_tf: X_hat[:,15:16], self.m150_tf: X_hat[:,16:17], self.m160_tf: X_hat[:,17:18], 
                                                self.m170_tf: X_hat[:,18:19], self.m180_tf: X_hat[:,19:20], self.m190_tf: X_hat[:,20:21], 
                                                self.m200_tf: X_hat[:,21:22], self.m210_tf: X_hat[:,22:23], self.m220_tf: X_hat[:,23:24],
                                                self.m230_tf: X_hat[:,24:25], self.m240_tf: X_hat[:,25:26], self.m250_tf: X_hat[:,26:27], 
                                                self.m260_tf: X_hat[:,27:28], self.m270_tf: X_hat[:,28:29], self.m280_tf: X_hat[:,29:30], 
                                                self.m290_tf: X_hat[:,30:31], self.m300_tf: X_hat[:,31:32], self.m310_tf: X_hat[:,32:33], 
                                                self.m320_tf: X_hat[:,33:34], self.m330_tf: X_hat[:,34:35], self.m340_tf: X_hat[:,35:36],
                                                self.m350_tf: X_hat[:,36:37], self.m360_tf: X_hat[:,37:38], self.m370_tf: X_hat[:,38:39],
                                                self.m380_tf: X_hat[:,39:40], self.m390_tf: X_hat[:,40:41], self.m400_tf: X_hat[:,41:42],
                                                self.m410_tf: X_hat[:,42:43], self.m420_tf: X_hat[:,43:44], self.m430_tf: X_hat[:,44:45], 
                                                self.m440_tf: X_hat[:,45:46], self.m450_tf: X_hat[:,46:47], self.m460_tf: X_hat[:,47:48],
                                                self.m470_tf: X_hat[:,48:49], self.m480_tf: X_hat[:,49:50], self.m490_tf: X_hat[:,50:51],
                                                self.m500_tf: X_hat[:,51:52]}) 
        qb_hat = self.sess.run(self.qb0_pred, {self.z0_tf: X_hat[:,0:1], self.t0_tf: X_hat[:,1:2], self.m10_tf: X_hat[:,2:3], 
                                                self.m20_tf: X_hat[:,3:4], self.m30_tf: X_hat[:,4:5], self.m40_tf: X_hat[:,5:6], 
                                                self.m50_tf: X_hat[:,6:7], self.m60_tf: X_hat[:,7:8], self.m70_tf: X_hat[:,8:9],
                                                self.m80_tf: X_hat[:,9:10], self.m90_tf: X_hat[:,10:11], self.m100_tf: X_hat[:,11:12],
                                                self.m110_tf: X_hat[:,12:13], self.m120_tf: X_hat[:,13:14], self.m130_tf: X_hat[:,14:15], 
                                                self.m140_tf: X_hat[:,15:16], self.m150_tf: X_hat[:,16:17], self.m160_tf: X_hat[:,17:18], 
                                                self.m170_tf: X_hat[:,18:19], self.m180_tf: X_hat[:,19:20], self.m190_tf: X_hat[:,20:21], 
                                                self.m200_tf: X_hat[:,21:22], self.m210_tf: X_hat[:,22:23], self.m220_tf: X_hat[:,23:24],
                                                self.m230_tf: X_hat[:,24:25], self.m240_tf: X_hat[:,25:26], self.m250_tf: X_hat[:,26:27], 
                                                self.m260_tf: X_hat[:,27:28], self.m270_tf: X_hat[:,28:29], self.m280_tf: X_hat[:,29:30], 
                                                self.m290_tf: X_hat[:,30:31], self.m300_tf: X_hat[:,31:32], self.m310_tf: X_hat[:,32:33], 
                                                self.m320_tf: X_hat[:,33:34], self.m330_tf: X_hat[:,34:35], self.m340_tf: X_hat[:,35:36],
                                                self.m350_tf: X_hat[:,36:37], self.m360_tf: X_hat[:,37:38], self.m370_tf: X_hat[:,38:39],
                                                self.m380_tf: X_hat[:,39:40], self.m390_tf: X_hat[:,40:41], self.m400_tf: X_hat[:,41:42],
                                                self.m410_tf: X_hat[:,42:43], self.m420_tf: X_hat[:,43:44], self.m430_tf: X_hat[:,44:45], 
                                                self.m440_tf: X_hat[:,45:46], self.m450_tf: X_hat[:,46:47], self.m460_tf: X_hat[:,47:48],
                                                self.m470_tf: X_hat[:,48:49], self.m480_tf: X_hat[:,49:50], self.m490_tf: X_hat[:,50:51],
                                                self.m500_tf: X_hat[:,51:52]})                                              
        f_y_hat = self.sess.run(self.f_y_pred, {self.z_c_tf: X_hat[:,0:1], self.t_c_tf: X_hat[:,1:2], self.m1_c_tf: X_hat[:,2:3], 
                                                self.m2_c_tf: X_hat[:,3:4], self.m3_c_tf: X_hat[:,4:5], self.m4_c_tf: X_hat[:,5:6], 
                                                self.m5_c_tf: X_hat[:,6:7], self.m6_c_tf: X_hat[:,7:8], self.m7_c_tf: X_hat[:,8:9],
                                                self.m8_c_tf: X_hat[:,9:10], self.m9_c_tf: X_hat[:,10:11], self.m10_c_tf: X_hat[:,11:12],
                                                self.m11_c_tf: X_hat[:,12:13], self.m12_c_tf: X_hat[:,13:14], self.m13_c_tf: X_hat[:,14:15], 
                                                self.m14_c_tf: X_hat[:,15:16], self.m15_c_tf: X_hat[:,16:17], self.m16_c_tf: X_hat[:,17:18], 
                                                self.m17_c_tf: X_hat[:,18:19], self.m18_c_tf: X_hat[:,19:20], self.m19_c_tf: X_hat[:,20:21], 
                                                self.m20_c_tf: X_hat[:,21:22], self.m21_c_tf: X_hat[:,22:23], self.m22_c_tf: X_hat[:,23:24],
                                                self.m23_c_tf: X_hat[:,24:25], self.m24_c_tf: X_hat[:,25:26], self.m25_c_tf: X_hat[:,26:27], 
                                                self.m26_c_tf: X_hat[:,27:28], self.m27_c_tf: X_hat[:,28:29], self.m28_c_tf: X_hat[:,29:30], 
                                                self.m29_c_tf: X_hat[:,30:31], self.m30_c_tf: X_hat[:,31:32], self.m31_c_tf: X_hat[:,32:33], 
                                                self.m32_c_tf: X_hat[:,33:34], self.m33_c_tf: X_hat[:,34:35], self.m34_c_tf: X_hat[:,35:36],
                                                self.m35_c_tf: X_hat[:,36:37], self.m36_c_tf: X_hat[:,37:38], self.m37_c_tf: X_hat[:,38:39],
                                                self.m38_c_tf: X_hat[:,39:40], self.m39_c_tf: X_hat[:,40:41], self.m40_c_tf: X_hat[:,41:42],
                                                self.m41_c_tf: X_hat[:,42:43], self.m42_c_tf: X_hat[:,43:44], self.m43_c_tf: X_hat[:,44:45], 
                                                self.m44_c_tf: X_hat[:,45:46], self.m45_c_tf: X_hat[:,46:47], self.m46_c_tf: X_hat[:,47:48],
                                                self.m47_c_tf: X_hat[:,48:49], self.m48_c_tf: X_hat[:,49:50], self.m49_c_tf: X_hat[:,50:51],
                                                self.m50_c_tf: X_hat[:,51:52]})
        f_p_hat = self.sess.run(self.f_p_pred, {self.z_c_tf: X_hat[:,0:1], self.t_c_tf: X_hat[:,1:2], self.m1_c_tf: X_hat[:,2:3], 
                                                self.m2_c_tf: X_hat[:,3:4], self.m3_c_tf: X_hat[:,4:5], self.m4_c_tf: X_hat[:,5:6], 
                                                self.m5_c_tf: X_hat[:,6:7], self.m6_c_tf: X_hat[:,7:8], self.m7_c_tf: X_hat[:,8:9],
                                                self.m8_c_tf: X_hat[:,9:10], self.m9_c_tf: X_hat[:,10:11], self.m10_c_tf: X_hat[:,11:12],
                                                self.m11_c_tf: X_hat[:,12:13], self.m12_c_tf: X_hat[:,13:14], self.m13_c_tf: X_hat[:,14:15], 
                                                self.m14_c_tf: X_hat[:,15:16], self.m15_c_tf: X_hat[:,16:17], self.m16_c_tf: X_hat[:,17:18], 
                                                self.m17_c_tf: X_hat[:,18:19], self.m18_c_tf: X_hat[:,19:20], self.m19_c_tf: X_hat[:,20:21], 
                                                self.m20_c_tf: X_hat[:,21:22], self.m21_c_tf: X_hat[:,22:23], self.m22_c_tf: X_hat[:,23:24],
                                                self.m23_c_tf: X_hat[:,24:25], self.m24_c_tf: X_hat[:,25:26], self.m25_c_tf: X_hat[:,26:27], 
                                                self.m26_c_tf: X_hat[:,27:28], self.m27_c_tf: X_hat[:,28:29], self.m28_c_tf: X_hat[:,29:30], 
                                                self.m29_c_tf: X_hat[:,30:31], self.m30_c_tf: X_hat[:,31:32], self.m31_c_tf: X_hat[:,32:33], 
                                                self.m32_c_tf: X_hat[:,33:34], self.m33_c_tf: X_hat[:,34:35], self.m34_c_tf: X_hat[:,35:36],
                                                self.m35_c_tf: X_hat[:,36:37], self.m36_c_tf: X_hat[:,37:38], self.m37_c_tf: X_hat[:,38:39],
                                                self.m38_c_tf: X_hat[:,39:40], self.m39_c_tf: X_hat[:,40:41], self.m40_c_tf: X_hat[:,41:42],
                                                self.m41_c_tf: X_hat[:,42:43], self.m42_c_tf: X_hat[:,43:44], self.m43_c_tf: X_hat[:,44:45], 
                                                self.m44_c_tf: X_hat[:,45:46], self.m45_c_tf: X_hat[:,46:47], self.m46_c_tf: X_hat[:,47:48],
                                                self.m47_c_tf: X_hat[:,48:49], self.m48_c_tf: X_hat[:,49:50], self.m49_c_tf: X_hat[:,50:51],
                                                self.m50_c_tf: X_hat[:,51:52]})

        w_hat = self.sess.run(self.weights, {self.z_c_tf: X_hat[:,0:1], self.t_c_tf: X_hat[:,1:2], self.m1_c_tf: X_hat[:,2:3], 
                                                self.m2_c_tf: X_hat[:,3:4], self.m3_c_tf: X_hat[:,4:5], self.m4_c_tf: X_hat[:,5:6], 
                                                self.m5_c_tf: X_hat[:,6:7], self.m6_c_tf: X_hat[:,7:8], self.m7_c_tf: X_hat[:,8:9],
                                                self.m8_c_tf: X_hat[:,9:10], self.m9_c_tf: X_hat[:,10:11], self.m10_c_tf: X_hat[:,11:12],
                                                self.m11_c_tf: X_hat[:,12:13], self.m12_c_tf: X_hat[:,13:14], self.m13_c_tf: X_hat[:,14:15], 
                                                self.m14_c_tf: X_hat[:,15:16], self.m15_c_tf: X_hat[:,16:17], self.m16_c_tf: X_hat[:,17:18], 
                                                self.m17_c_tf: X_hat[:,18:19], self.m18_c_tf: X_hat[:,19:20], self.m19_c_tf: X_hat[:,20:21], 
                                                self.m20_c_tf: X_hat[:,21:22], self.m21_c_tf: X_hat[:,22:23], self.m22_c_tf: X_hat[:,23:24],
                                                self.m23_c_tf: X_hat[:,24:25], self.m24_c_tf: X_hat[:,25:26], self.m25_c_tf: X_hat[:,26:27], 
                                                self.m26_c_tf: X_hat[:,27:28], self.m27_c_tf: X_hat[:,28:29], self.m28_c_tf: X_hat[:,29:30], 
                                                self.m29_c_tf: X_hat[:,30:31], self.m30_c_tf: X_hat[:,31:32], self.m31_c_tf: X_hat[:,32:33], 
                                                self.m32_c_tf: X_hat[:,33:34], self.m33_c_tf: X_hat[:,34:35], self.m34_c_tf: X_hat[:,35:36],
                                                self.m35_c_tf: X_hat[:,36:37], self.m36_c_tf: X_hat[:,37:38], self.m37_c_tf: X_hat[:,38:39],
                                                self.m38_c_tf: X_hat[:,39:40], self.m39_c_tf: X_hat[:,40:41], self.m40_c_tf: X_hat[:,41:42],
                                                self.m41_c_tf: X_hat[:,42:43], self.m42_c_tf: X_hat[:,43:44], self.m43_c_tf: X_hat[:,44:45], 
                                                self.m44_c_tf: X_hat[:,45:46], self.m45_c_tf: X_hat[:,46:47], self.m46_c_tf: X_hat[:,47:48],
                                                self.m47_c_tf: X_hat[:,48:49], self.m48_c_tf: X_hat[:,49:50], self.m49_c_tf: X_hat[:,50:51],
                                                self.m50_c_tf: X_hat[:,51:52]})
        
        b_hat = self.sess.run(self.biases, {self.z_c_tf: X_hat[:,0:1], self.t_c_tf: X_hat[:,1:2], self.m1_c_tf: X_hat[:,2:3], 
                                                self.m2_c_tf: X_hat[:,3:4], self.m3_c_tf: X_hat[:,4:5], self.m4_c_tf: X_hat[:,5:6], 
                                                self.m5_c_tf: X_hat[:,6:7], self.m6_c_tf: X_hat[:,7:8], self.m7_c_tf: X_hat[:,8:9],
                                                self.m8_c_tf: X_hat[:,9:10], self.m9_c_tf: X_hat[:,10:11], self.m10_c_tf: X_hat[:,11:12],
                                                self.m11_c_tf: X_hat[:,12:13], self.m12_c_tf: X_hat[:,13:14], self.m13_c_tf: X_hat[:,14:15], 
                                                self.m14_c_tf: X_hat[:,15:16], self.m15_c_tf: X_hat[:,16:17], self.m16_c_tf: X_hat[:,17:18], 
                                                self.m17_c_tf: X_hat[:,18:19], self.m18_c_tf: X_hat[:,19:20], self.m19_c_tf: X_hat[:,20:21], 
                                                self.m20_c_tf: X_hat[:,21:22], self.m21_c_tf: X_hat[:,22:23], self.m22_c_tf: X_hat[:,23:24],
                                                self.m23_c_tf: X_hat[:,24:25], self.m24_c_tf: X_hat[:,25:26], self.m25_c_tf: X_hat[:,26:27], 
                                                self.m26_c_tf: X_hat[:,27:28], self.m27_c_tf: X_hat[:,28:29], self.m28_c_tf: X_hat[:,29:30], 
                                                self.m29_c_tf: X_hat[:,30:31], self.m30_c_tf: X_hat[:,31:32], self.m31_c_tf: X_hat[:,32:33], 
                                                self.m32_c_tf: X_hat[:,33:34], self.m33_c_tf: X_hat[:,34:35], self.m34_c_tf: X_hat[:,35:36],
                                                self.m35_c_tf: X_hat[:,36:37], self.m36_c_tf: X_hat[:,37:38], self.m37_c_tf: X_hat[:,38:39],
                                                self.m38_c_tf: X_hat[:,39:40], self.m39_c_tf: X_hat[:,40:41], self.m40_c_tf: X_hat[:,41:42],
                                                self.m41_c_tf: X_hat[:,42:43], self.m42_c_tf: X_hat[:,43:44], self.m43_c_tf: X_hat[:,44:45], 
                                                self.m44_c_tf: X_hat[:,45:46], self.m45_c_tf: X_hat[:,46:47], self.m46_c_tf: X_hat[:,47:48],
                                                self.m47_c_tf: X_hat[:,48:49], self.m48_c_tf: X_hat[:,49:50], self.m49_c_tf: X_hat[:,50:51],
                                                self.m50_c_tf: X_hat[:,51:52]})
               
        return y_hat, p_hat, qa_hat, qb_hat, f_y_hat, f_p_hat, w_hat, b_hat

if __name__ == "__main__": 
    
    #Load data from MATLAB
    data = scipy.io.loadmat('train_ads.mat')
    Z = np.real(data['Z'])
    T = np.real(data['T'])
    X0 = np.real(data['X0'])
    X_en = np.real(data['X_en'])
    X_lb = np.real(data['X_lb'])
    X_rb = np.real(data['X_rb'])
    X_c_train = np.real(data['X_c_train'])
    in_train = np.real(data['in_train'])
    en_train = np.real(data['en_train'])
    lb_train = np.real(data['lb_train'])
    rb_train = np.real(data['rb_train']) 
    low_bound = np.real(data['low_bound'])
    up_bound = np.real(data['up_bound'])
    coeff = np.real(data['coeff']).T #PDE coefficients
    N0_ids = np.real(data['N0_ids'])
    N_b_ids = np.real(data['N_b_ids'])
    N_c_ids = np.real(data['N_c_ids'])
    X_sol_1 = np.real(data['X_sol_1'])
    y_sol_1 = np.real(data['y_sol_1'])
    p_sol_1 = np.real(data['p_sol_1'])
    qa_sol_1 = np.real(data['qa_sol_1'])
    qb_sol_1 = np.real(data['qb_sol_1'])
   
    #Neural network architecture
    layers = [52, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 4]

  
    
    #Index IDs
    N0_ids = N0_ids.reshape(-1)
    N_b_ids = N_b_ids.reshape(-1)
    N_c_ids = N_c_ids.reshape(-1)
    N_k = 60 #number of initial profiles

    model = PANACHE(X0, X_en, X_lb, X_rb, in_train, en_train, lb_train, rb_train, 
                              X_c_train, layers, low_bound, up_bound, coeff, N0_ids, N_b_ids, N_c_ids, N_k)
    
    #Train the model
    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    #Save the model
    model.save()
    
    #Predict the solutions of any one case (here case#1)
    y_hat, p_hat, qa_hat, qb_hat, f_y_hat, f_p_hat, w_hat, b_hat = model.predict(X_sol_1)
            
    error_y = np.linalg.norm(y_sol_1-y_hat,2)/np.linalg.norm(y_sol_1,2)
    print('Error y: %e' % (error_y))     

    error_p = np.linalg.norm(p_sol_1-p_hat,2)/np.linalg.norm(p_sol_1,2)
    print('Error p: %e' % (error_p))

    error_qa = np.linalg.norm(qa_sol_1-qa_hat,2)/np.linalg.norm(qa_sol_1,2)
    print('Error qa: %e' % (error_qa))

    error_qb = np.linalg.norm(qb_sol_1-qb_hat,2)/np.linalg.norm(qb_sol_1,2)
    print('Error qb: %e' % (error_qb))
    
    U_pred_1 = griddata(X_sol_1[:,:2], y_hat.flatten(), (Z, T), method='cubic')
    P_pred_1 = griddata(X_sol_1[:,:2], p_hat.flatten(), (Z, T), method='cubic')
    Qa_pred_1 = griddata(X_sol_1[:,:2], qa_hat.flatten(), (Z, T), method='cubic')
    Qb_pred_1 = griddata(X_sol_1[:,:2], qb_hat.flatten(), (Z, T), method='cubic')
    
    b_pred = np.empty((len(b_hat),), dtype=np.object)
    for i in range(len(b_hat)):
        b_pred[i] = b_hat[i]
    
    scipy.io.savemat('workspace.mat',{'U_pred_1': U_pred_1, 'P_pred_1': P_pred_1, \
                                       'Qa_pred_1': Qa_pred_1, 'Qb_pred_1': Qb_pred_1,  \
                                       'w_pred': w_hat, 'b_pred':b_pred})