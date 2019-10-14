# import plotting packages
from os.path import join
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap,  Colormap

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

import rasterio.mask
from shapely.geometry import box, mapping, LineString, Point
import geopandas as gp

 
# Author: Anton Mikhailov; https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
google_turbo_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
cmap_turbo=ListedColormap(google_turbo_data)
# https://github.com/matplotlib/cmocean/blob/master/cmocean/rgb/phase.py
cmocean_phase_data = [[ 0.65830839, 0.46993917, 0.04941288],[ 0.66433742, 0.4662019 , 0.05766473],[ 0.67020869, 0.46248014, 0.0653456 ],[ 0.67604299, 0.45869838, 0.07273174],[ 0.68175228, 0.45491407, 0.07979262],[ 0.6874028 , 0.45108417, 0.08667103],[ 0.6929505 , 0.44723893, 0.09335869],[ 0.69842619, 0.44335768, 0.09992839],[ 0.7038123 , 0.43945328, 0.1063871 ],[ 0.70912069, 0.43551765, 0.11277174],[ 0.71434524, 0.43155576, 0.11909348],[ 0.71949289, 0.42756272, 0.12537606],[ 0.72455619, 0.4235447 , 0.13162325],[ 0.72954895, 0.41949098, 0.13786305],[ 0.73445172, 0.41541774, 0.14408039],[ 0.73929496, 0.41129973, 0.15032217],[ 0.74403834, 0.40717158, 0.15654335],[ 0.74873695, 0.40298519, 0.16282282],[ 0.75332319, 0.39880107, 0.16907566],[ 0.75788083, 0.39454245, 0.17542179],[ 0.7623326 , 0.39028096, 0.18175915],[ 0.76673205, 0.38596549, 0.18816819],[ 0.77105247, 0.38162141, 0.19461532],[ 0.77529528, 0.37724732, 0.20110652],[ 0.77948666, 0.37281509, 0.2076873 ],[ 0.78358534, 0.36836772, 0.21429736],[ 0.78763763, 0.363854  , 0.22101648],[ 0.79161134, 0.35930804, 0.2277974 ],[ 0.79550606, 0.3547299 , 0.23464353],[ 0.79935398, 0.35007959, 0.24161832],[ 0.80311671, 0.34540152, 0.24865892],[ 0.80681033, 0.34067452, 0.25580075],[ 0.8104452 , 0.33588248, 0.26307222],[ 0.8139968 , 0.33105538, 0.27043183],[ 0.81747689, 0.32617526, 0.27791096],[ 0.82089415, 0.32122629, 0.28553846],[ 0.82422713, 0.3162362 , 0.29327617],[ 0.82747661, 0.31120154, 0.30113388],[ 0.83066399, 0.30608459, 0.30917579],[ 0.83376307, 0.30092244, 0.31734921],[ 0.83677286, 0.29571346, 0.32566199],[ 0.83969693, 0.29044723, 0.33413665],[ 0.84253873, 0.28511151, 0.34279962],[ 0.84528297, 0.27972917, 0.35162078],[ 0.84792704, 0.27430045, 0.36060681],[ 0.85046793, 0.26882624, 0.36976395],[ 0.85291056, 0.26328859, 0.37913116],[ 0.855242  , 0.25770888, 0.38868217],[ 0.85745673, 0.25209367, 0.39841601],[ 0.85955023, 0.24644737, 0.40833625],[ 0.86151767, 0.24077563, 0.41844557],[ 0.86335392, 0.23508521, 0.42874606],[ 0.86505685, 0.22937288, 0.43926008],[ 0.86661606, 0.22366308, 0.44996127],[ 0.86802578, 0.21796785, 0.46084758],[ 0.86928003, 0.21230132, 0.47191554],[ 0.87037274, 0.20667988, 0.48316015],[ 0.87129781, 0.2011224 , 0.49457479],[ 0.87204914, 0.19565041, 0.50615118],[ 0.87262076, 0.19028829, 0.51787932],[ 0.87300686, 0.18506334, 0.5297475 ],[ 0.8732019 , 0.18000588, 0.54174232],[ 0.87320066, 0.1751492 , 0.55384874],[ 0.87299833, 0.17052942, 0.56605016],[ 0.87259058, 0.16618514, 0.57832856],[ 0.87197361, 0.16215698, 0.59066466],[ 0.87114414, 0.15848667, 0.60303881],[ 0.87009966, 0.15521687, 0.61542844],[ 0.86883823, 0.15238892, 0.62781175],[ 0.86735858, 0.15004199, 0.64016651],[ 0.8656601 , 0.14821149, 0.65247022],[ 0.86374282, 0.14692762, 0.66470043],[ 0.86160744, 0.14621386, 0.67683495],[ 0.85925523, 0.14608582, 0.68885204],[ 0.85668805, 0.14655046, 0.70073065],[ 0.85390829, 0.14760576, 0.71245054],[ 0.85091881, 0.14924094, 0.7239925 ],[ 0.84772287, 0.15143717, 0.73533849],[ 0.84432409, 0.15416865, 0.74647174],[ 0.84072639, 0.15740403, 0.75737678],[ 0.83693394, 0.16110786, 0.76803952],[ 0.83295108, 0.16524205, 0.77844723],[ 0.82878232, 0.16976729, 0.78858858],[ 0.82443225, 0.17464414, 0.7984536 ],[ 0.81990551, 0.179834  , 0.80803365],[ 0.81520674, 0.18529984, 0.8173214 ],[ 0.81034059, 0.19100664, 0.82631073],[ 0.80531176, 0.1969216 , 0.83499645],[ 0.80012467, 0.20301465, 0.84337486],[ 0.79478367, 0.20925826, 0.8514432 ],[ 0.78929302, 0.21562737, 0.85919957],[ 0.78365681, 0.22209936, 0.86664294],[ 0.77787898, 0.22865386, 0.87377308],[ 0.7719633 , 0.23527265, 0.88059043],[ 0.76591335, 0.24193947, 0.88709606],[ 0.7597325 , 0.24863985, 0.89329158],[ 0.75342394, 0.25536094, 0.89917908],[ 0.74699063, 0.26209137, 0.90476105],[ 0.74043533, 0.2688211 , 0.91004033],[ 0.73376055, 0.27554128, 0.91502   ],[ 0.72696862, 0.28224415, 0.91970339],[ 0.7200616 , 0.2889229 , 0.92409395],[ 0.71304134, 0.29557159, 0.92819525],[ 0.70590945, 0.30218508, 0.9320109 ],[ 0.69866732, 0.30875887, 0.93554451],[ 0.69131609, 0.31528914, 0.93879964],[ 0.68385669, 0.32177259, 0.94177976],[ 0.6762898 , 0.32820641, 0.94448822],[ 0.6686159 , 0.33458824, 0.94692818],[ 0.66083524, 0.3409161 , 0.94910264],[ 0.65294785, 0.34718834, 0.95101432],[ 0.64495358, 0.35340362, 0.95266571],[ 0.63685208, 0.35956083, 0.954059  ],[ 0.62864284, 0.3656591 , 0.95519608],[ 0.62032517, 0.3716977 , 0.95607853],[ 0.61189825, 0.37767607, 0.95670757],[ 0.60336117, 0.38359374, 0.95708408],[ 0.59471291, 0.3894503 , 0.95720861],[ 0.58595242, 0.39524541, 0.95708134],[ 0.5770786 , 0.40097871, 0.95670212],[ 0.56809041, 0.40664983, 0.95607045],[ 0.55898686, 0.41225834, 0.95518556],[ 0.54976709, 0.41780374, 0.95404636],[ 0.5404304 , 0.42328541, 0.95265153],[ 0.53097635, 0.42870263, 0.95099953],[ 0.52140479, 0.43405447, 0.94908866],[ 0.51171597, 0.43933988, 0.94691713],[ 0.50191056, 0.44455757, 0.94448311],[ 0.49198981, 0.44970607, 0.94178481],[ 0.48195555, 0.45478367, 0.93882055],[ 0.47181035, 0.45978843, 0.93558888],[ 0.46155756, 0.46471821, 0.93208866],[ 0.45119801, 0.46957218, 0.92831786],[ 0.44073852, 0.47434688, 0.92427669],[ 0.43018722, 0.47903864, 0.9199662 ],[ 0.41955166, 0.4836444 , 0.91538759],[ 0.40884063, 0.48816094, 0.91054293],[ 0.39806421, 0.49258494, 0.90543523],[ 0.38723377, 0.49691301, 0.90006852],[ 0.37636206, 0.50114173, 0.89444794],[ 0.36546127, 0.5052684 , 0.88857877],[ 0.35454654, 0.5092898 , 0.88246819],[ 0.34363779, 0.51320158, 0.87612664],[ 0.33275309, 0.51700082, 0.86956409],[ 0.32191166, 0.52068487, 0.86279166],[ 0.31113372, 0.52425144, 0.85582152],[ 0.3004404 , 0.52769862, 0.84866679],[ 0.28985326, 0.53102505, 0.84134123],[ 0.27939616, 0.53422931, 0.83386051],[ 0.26909181, 0.53731099, 0.82623984],[ 0.258963  , 0.5402702 , 0.81849475],[ 0.24903239, 0.54310763, 0.8106409 ],[ 0.23932229, 0.54582448, 0.80269392],[ 0.22985664, 0.54842189, 0.79467122],[ 0.2206551 , 0.55090241, 0.78658706],[ 0.21173641, 0.55326901, 0.77845533],[ 0.20311843, 0.55552489, 0.77028973],[ 0.1948172 , 0.55767365, 0.76210318],[ 0.1868466 , 0.55971922, 0.75390763],[ 0.17921799, 0.56166586, 0.74571407],[ 0.1719422 , 0.56351747, 0.73753498],[ 0.16502295, 0.56527915, 0.72937754],[ 0.15846116, 0.566956  , 0.72124819],[ 0.15225499, 0.56855297, 0.71315321],[ 0.14639876, 0.57007506, 0.70509769],[ 0.14088284, 0.57152729, 0.69708554],[ 0.13569366, 0.57291467, 0.68911948],[ 0.13081385, 0.57424211, 0.68120108],[ 0.12622247, 0.57551447, 0.67333078],[ 0.12189539, 0.57673644, 0.66550792],[ 0.11780654, 0.57791235, 0.65773233],[ 0.11392613, 0.5790468 , 0.64999984],[ 0.11022348, 0.58014398, 0.64230637],[ 0.10666732, 0.58120782, 0.63464733],[ 0.10322631, 0.58224198, 0.62701729],[ 0.0998697 , 0.58324982, 0.61941001],[ 0.09656813, 0.58423445, 0.61181853],[ 0.09329429, 0.58519864, 0.60423523],[ 0.09002364, 0.58614483, 0.5966519 ],[ 0.08673514, 0.58707512, 0.58905979],[ 0.08341199, 0.58799127, 0.58144971],[ 0.08004245, 0.58889466, 0.57381211],[ 0.07662083, 0.58978633, 0.56613714],[ 0.07314852, 0.59066692, 0.55841474],[ 0.06963541, 0.5915367 , 0.55063471],[ 0.06610144, 0.59239556, 0.54278681],[ 0.06257861, 0.59324304, 0.53486082],[ 0.05911304, 0.59407833, 0.52684614],[ 0.05576765, 0.5949003 , 0.5187322 ],[ 0.05262511, 0.59570732, 0.51050978],[ 0.04978881, 0.5964975 , 0.50216936],[ 0.04738319, 0.59726862, 0.49370174],[ 0.04555067, 0.59801813, 0.48509809],[ 0.04444396, 0.59874316, 0.47635   ],[ 0.04421323, 0.59944056, 0.46744951],[ 0.04498918, 0.60010687, 0.45838913],[ 0.04686604, 0.60073837, 0.44916187],[ 0.04988979, 0.60133103, 0.43976125],[ 0.05405573, 0.60188055, 0.4301812 ],[ 0.05932209, 0.60238289, 0.42040543],[ 0.06560774, 0.60283258, 0.41043772],[ 0.07281962, 0.60322442, 0.40027363],[ 0.08086177, 0.60355283, 0.38990941],[ 0.08964366, 0.60381194, 0.37934208],[ 0.09908952, 0.60399554, 0.36856412],[ 0.10914617, 0.60409695, 0.35755799],[ 0.11974119, 0.60410858, 0.34634096],[ 0.13082746, 0.6040228 , 0.33491416],[ 0.14238003, 0.60383119, 0.323267  ],[ 0.1543847 , 0.60352425, 0.31138823],[ 0.16679093, 0.60309301, 0.29931029],[ 0.17959757, 0.60252668, 0.2870237 ],[ 0.19279966, 0.60181364, 0.27452964],[ 0.20634465, 0.60094466, 0.2618794 ],[ 0.22027287, 0.5999043 , 0.24904251],[ 0.23449833, 0.59868591, 0.23611022],[ 0.24904416, 0.5972746 , 0.2230778 ],[ 0.26382006, 0.59566656, 0.21004673],[ 0.2788104 , 0.5938521 , 0.19705484],[ 0.29391494, 0.59183348, 0.18421621],[ 0.3090634 , 0.58961302, 0.17161942],[ 0.32415577, 0.58720132, 0.15937753],[ 0.3391059 , 0.58461164, 0.14759012],[ 0.35379624, 0.58186793, 0.13637734],[ 0.36817905, 0.5789861 , 0.12580054],[ 0.38215966, 0.57599512, 0.1159504 ],[ 0.39572824, 0.57290928, 0.10685038],[ 0.40881926, 0.56975727, 0.09855521],[ 0.42148106, 0.56654159, 0.09104002],[ 0.43364953, 0.56329296, 0.08434116],[ 0.44538908, 0.56000859, 0.07841305],[ 0.45672421, 0.5566943 , 0.07322913],[ 0.46765017, 0.55336373, 0.06876762],[ 0.47819138, 0.5500213 , 0.06498436],[ 0.48839686, 0.54666195, 0.06182163],[ 0.49828924, 0.5432874 , 0.05922726],[ 0.50789114, 0.53989827, 0.05714466],[ 0.51722475, 0.53649429, 0.05551476],[ 0.5263115 , 0.53307443, 0.05427793],[ 0.53517186, 0.52963707, 0.05337567],[ 0.54382515, 0.52618009, 0.05275208],[ 0.55228947, 0.52270103, 0.05235479],[ 0.56058163, 0.51919713, 0.0521356 ],[ 0.56871719, 0.51566545, 0.05205062],[ 0.57671045, 0.51210292, 0.0520602 ],[ 0.5845745 , 0.50850636, 0.05212851],[ 0.59232129, 0.50487256, 0.05222299],[ 0.5999617 , 0.50119827, 0.05231367],[ 0.60750568, 0.49748022, 0.05237234],[ 0.61496232, 0.49371512, 0.05237168],[ 0.62233999, 0.48989963, 0.05228423],[ 0.62964652, 0.48603032, 0.05208127],[ 0.63688935, 0.48210362, 0.05173155],[ 0.64407572, 0.4781157 , 0.0511996 ],[ 0.65121289, 0.47406244, 0.05044367],[ 0.65830839, 0.46993917, 0.04941288]]
cmap_phase=ListedColormap(cmocean_phase_data)


def basemap(axg, bbox=None, features=['land', 'rivers'], feat_colors=None, scale='50m', 
        gridlines=False, gridspacing=30, outline=True):
    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox 
        axg.set_extent((xmin, xmax, ymax, ymin), crs=ccrs.PlateCarree())
        if gridlines:
            grid_ymin = (np.floor(ymin/gridspacing)+1) * gridspacing
            grid_xmin = (np.floor(xmin/gridspacing)+1) * gridspacing
            lats = np.arange(grid_ymin, ymax, gridspacing)
            lons = np.arange(grid_xmin, xmax, gridspacing)
            set_gridlines(axg, xmin, ymin, lats, lons)
    elif gridlines:
        set_gridlines(axg)
    # features
    for i, f in enumerate(features):
        if feat_colors:
            kwargs = dict(color=feat_colors[i])
        else:
            kwargs = {}
        try:
            axg.add_feature(getattr(cfeature, f.upper()).with_scale(scale), zorder=0, **kwargs)
        except Exception as e:
            print(str(e))
    # plot borders in white
    axg.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='w', zorder=0)
    axg.outline_patch.set_visible(outline)

def add_regions(axg, regions, crs=ccrs.PlateCarree(), dx=0, dy=0.5, **kwargs):
    for name, bbox in regions.items():
        xmin, ymax = bbox[0], bbox[3]
        axg.add_geometries([box(*bbox)], crs=crs, facecolor='none', edgecolor=(0.5, 0.5, 0.5, 0.8), linewidth=0.8)
        axg.text(xmin+dx, ymax+dy, name, transform=crs, **kwargs) # annotation

def set_gridlines(axg, xmin=-180, ymin=-90, lats = np.arange(-60, 61, 30), lons = np.arange(-150, 151, 30)):
    gp_lats = gp.GeoDataFrame(
        geometry = [LineString([[l, lat] for l in np.linspace(-180, 180, 361)]) for lat in lats] + 
                    [LineString([[lon, l] for l in np.linspace(-90, 90, 181)]) for lon in lons],
        crs = {'a': 57.29577951308232, 'proj': 'eqc', 'lon_0': 0.0}
    )
    gp_lats.to_crs(axg.projection.proj4_init).plot(ax=axg, linestyle='--', linewidth=0.5, color='k', alpha=0.5, zorder=-1)
    for lat in lats:
        axg.text(xmin, lat, LATITUDE_FORMATTER(lat), transform=ccrs.PlateCarree(), 
                horizontalalignment='right', verticalalignment='bottom' if lat >=0 else 'top')
    for lon in lons:
        axg.text(lon, ymin, LONGITUDE_FORMATTER(lon), transform=ccrs.PlateCarree(), 
                horizontalalignment='center', verticalalignment='top')

# raster
def plot_gtiff_fromfile(ax, fn, bbox, cmap=cmap_turbo, nozero=True, vmin=None, vmax=None, cbar=False, cbar_kwargs={}, **kwargs):
    bbox_feat = [mapping(box(*bbox))]
    with rasterio.open(fn, 'r') as src:
        data, transform = rasterio.mask.mask(src, bbox_feat, crop=True)
        if nozero:
            data = np.where(data==0, src.nodata, data)
        data = np.ma.masked_values(data[0, :, :], src.nodata)
    im, cbar = plot_gtiff(ax, data, transform, cmap=cmap, vmin=vmin, vmax=vmax, 
                          cbar=cbar, cbar_kwargs=cbar_kwargs, **kwargs)
    return im, cbar

def plot_gtiff(ax, data, transform, cmap=cmap_turbo, vmin=None, vmax=None, cbar=False, cbar_kwargs={}, **kwargs):
    extent = plot_extent(data, transform)
    im = ax.imshow(data, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, origin='upper', **kwargs)
    ax.set_extent(extent)
    
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    vmin = vmin if vmin is not None else dmin 
    vmax = vmax if vmax is not None else dmax
    if (dmax > vmax) and (dmin < vmin):
        cbar_kwargs.update(extend='both')
    elif dmax > vmax:
        cbar_kwargs.update(extend='max')
    elif dmin < vmin:
        cbar_kwargs.update(extend='min')

    if cbar:
        # TODO: update according to plot_choropleth cbar
        cbar = set_colorbar(ax, im, **cbar_kwargs)
    else:
        cbar = None
    return im, cbar

def plot_extent(data, transform):
    rows, cols = data.shape[-2:]
    left, top = transform * (0, 0)
    right, bottom = transform * (cols, rows)
    extent = (left, right, bottom, top)
    return extent

# plot vector with scale
def pandas2geopandas(df, x_col='lon', y_col='lat', crs=ccrs.PlateCarree.proj4_init):
    geoms = [Point(x, y) for x, y in zip(df[x_col], df[y_col])]
    return gp.GeoDataFrame(df.drop([x_col, y_col], axis=1), geometry=geoms, crs=crs)

def plot_choropleth(figl, axg, gdf, column, 
                    cmap=cmap_turbo, vmin=None, vmax=None, norm=None, cticks=None, discrete=False,
                    plot_kwargs={}, cbar_kwargs={}, cbar_pos={}):
    
    # colormap 
    dmin, dmax = np.nanmin(gdf.loc[:, column].values), np.nanmax(gdf.loc[:, column].values)
    vmin = vmin if (vmin is not None) else dmin 
    vmax = vmax if (vmax is not None) else dmax
    if discrete: #only linear
        if cticks is None:
            k = cbar_kwargs.get('k', 10)
            cticks = np.linspace(vmin, vmax, k).tolist()
        norm = BoundaryNorm(cticks, cmap.N)
    elif norm is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = norm(vmin=vmin, vmax=vmax)
    if isinstance(cmap, str):
        cmap = Colormap(cmap)
    
    # add choropleth layer to fig
    gdf.plot(ax=axg, column=column, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, **plot_kwargs)

    # plot colorbar
    cax = None
    if cbar_kwargs:
        location = cbar_kwargs.pop('location', 'bottom')
        clabel = cbar_kwargs.pop('label', '')
        cbar_kwargs.update(ticks=cticks, boundaries=cticks if discrete else None)
        if 'extend' not in cbar_kwargs:
            if (dmax > vmax) and (dmin < vmin):
                cbar_kwargs.update(extend='both')
            elif dmax > vmax:
                cbar_kwargs.update(extend='max')
            elif dmin < vmin:
                cbar_kwargs.update(extend='min')
        if location == 'right':
            cax, kw = mpl.colorbar.make_axes_gridspec(axg, orientation='vertical', **cbar_pos)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, **cbar_kwargs)
            cbar.ax.set_ylabel(clabel, rotation='vertical')
            
            cpos = cax.get_position()
            gpos = axg.get_position()
            cax.set_position([cpos.x0, gpos.y0, cpos.width, gpos.height])

        elif location == 'bottom':
            cax, kw = mpl.colorbar.make_axes_gridspec(axg, orientation='horizontal', **cbar_pos)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal', **cbar_kwargs)
            cbar.ax.set_xlabel(clabel)
        else:
            raise ValueError('unknown value for "location"')
    return cax

# utils
def set_colorbar(ax, im, wspace=0.01, width=0.01, rel_height=0.5, label='', extend='max', inset_bottom=True):
    # make colorbar
    fig = plt.gcf()
    cax = fig.add_axes([1, 1, 0.1, 0.1]) # new ax
    cbar = fig.colorbar(im, extend=extend, cax=cax)
    cbar.ax.set_ylabel(label, rotation='vertical')
    posn = ax.get_position()
    h = posn.height * rel_height
    y0 = posn.y0 + posn.height - h
    if inset_bottom:
        y0 = posn.y0
    cax.set_position([posn.x0 + posn.width + wspace, y0, width, h])
    return cbar

def reset_geo_bounds(ax, bounds=None):
    if not bounds:
        bounds = ax.projection.domain.bounds
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])