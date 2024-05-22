"""Look-up table for metranet 8-bit values"""""
import numpy as np

METRANET_RR_8BIT = np.array([
    0.,    0.01    ,    0.02    ,    0.109569,    0.148698,
    0.189207,    0.231144,    0.274561,    0.319508,    0.36604 ,
    0.414214,    0.464086,    0.515717,    0.569168,    0.624505,
    0.681793,    0.741101,    0.802501,    0.866066,    0.931873,
    1.      ,    1.07053 ,    1.143547,    1.219139,    1.297397,
    1.378414,    1.462289,    1.549121,    1.639016,    1.73208 ,
    1.828427,    1.928171,    2.031433,    2.138336,    2.24901 ,
    2.363586,    2.482202,    2.605002,    2.732132,    2.863745,
    3.      ,    3.14106 ,    3.287094,    3.438278,    3.594793,
    3.756828,    3.924578,    4.098242,    4.278032,    4.464161,
    4.656854,    4.856343,    5.062866,    5.276673,    5.498019,
    5.727171,    5.964405,    6.210004,    6.464264,    6.72749 ,
    7.      ,    7.28212 ,    7.574187,    7.876555,    8.189587,
    8.513657,    8.849155,    9.196485,    9.556064,    9.928322,
    10.313708,   10.712686,   11.125732,   11.553346,   11.996038,
    12.454343,   12.928809,   13.420008,   13.928528,   14.454981,
    15.      ,   15.56424 ,   16.148375,   16.75311 ,   17.379173,
    18.027313,   18.69831 ,   19.39297 ,   20.112127,   20.856644,
    21.627417,   22.425371,   23.251465,   24.106691,   24.992077,
    25.908686,   26.857618,   27.840015,   28.857056,   29.909962,
    31.      ,   32.12848 ,   33.29675 ,   34.50622 ,   35.758347,
    37.054626,   38.39662 ,   39.78594 ,   41.224255,   42.713287,
    44.254833,   45.850742,   47.50293 ,   49.213383,   50.984154,
    52.81737 ,   54.715237,   56.68003 ,   58.71411 ,   60.819923,
    63.      ,   65.25696 ,   67.5935  ,   70.01244 ,   72.51669 ,
    75.10925 ,   77.79324 ,   80.57188 ,   83.44851 ,   86.426575,
    89.50967 ,   92.701485,   96.00586 ,   99.426765,  102.96831 ,
    106.63474 ,  110.43047 ,  114.36006 ,  118.42822 ,  122.63985 ,
    127.      ,  131.51392 ,  136.187   ,  141.02489 ,  146.03339 ,
    151.2185  ,  156.58649 ,  162.14375 ,  167.89702 ,  173.85315 ,
    180.01933 ,  186.40297 ,  193.01172 ,  199.85353 ,  206.93661 ,
    214.26949 ,  221.86095 ,  229.72012 ,  237.85645 ,  246.2797  ,
    255.      ,  264.02783 ,  273.374   ,  283.04977 ,  293.06677 ,
    303.437   ,  314.17297 ,  325.2875  ,  336.79404 ,  348.7063  ,
    361.03867 ,  373.80594 ,  387.02344 ,  400.70706 ,  414.87323 ,
    429.53897 ,  444.7219  ,  460.44025 ,  476.7129  ,  493.5594  ,
    511.      ,  529.05566 ,  547.748   ,  567.09955 ,  587.13354 ,
    607.874   ,  629.34595 ,  651.575   ,  674.5881  ,  698.4126  ,
    723.07733 ,  748.6119  ,  775.0469  ,  802.4141  ,  830.74646 ,
    860.07794 ,  890.4438  ,  921.8805  ,  954.4258  ,  988.1188  ,
    1023.      , 1059.1113  , 1096.496   , 1135.1991  , 1175.2671  ,
    1216.748   , 1259.6919  , 1304.15    , 1350.1761  , 1397.8252  ,
    1447.1547  , 1498.2238  , 1551.0938  , 1605.8282  , 1662.4929  ,
    1721.1559  , 1781.8876  , 1844.761   , 1909.8516  , 1977.2375  ,
    2047.      , 2119.2227  , 2193.992   , 2271.3982  , 2351.5342  ,
    2434.496   , 2520.3838  , 2609.3     , 2701.3523  , 2796.6504  ,
    2895.3093  , 2997.4475  , 3103.1875  , 3212.6565  , 3325.9858  ,
    3443.3118  , 3564.7751  , 3690.522   , 3820.7031  , 3955.475   ,
    4095.      , 4239.4453  , 4388.984   , 4543.7964  , 4704.0684  ,
    4869.992   , 5041.7676  , 5219.6     , 5403.7046  , 5594.301   ,
    5791.6187  ,
])