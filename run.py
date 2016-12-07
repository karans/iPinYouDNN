# def SGD(self, training_data, epochs, mini_batch_size, eta, validation_data, test_data, training_outputs, weight_file = None, variable_updates = False, log_file=None, lmbda=0.0):
# from theano.tensor.nnet import sigmoid
# from theano.tensor import tanh
import network3, pickle
from network3 import Network, ReLU
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer


p = [   2,    5,    9,   12,   15,   18,   21,   24,  260,  263,  266,  269,
  272,  275,  278,  457,  460,  463,  466,  470,  473,  476,  479,  482,
  485,  488,  491,  494,  497,  500,  503,  506,  509,  512,    0,    1,
    7,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   36,
   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,
   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,
   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,   72,
   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,
   85,   86,   87,   88,   89,   90,   91,    3,    6,   10,   13,   16,
   19,   22,   25,  261,  264,  267,  270,  273,  276,  279,  458,  461,
  464,  468,  471,  474,  477,  480,  483,  486,  489,  492,  495,  498,
  501,  504,  507,  510,  513,   92,   93,   94,   95,   96,   97,   98,
   99,  100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,
  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,
  123,  124,  125,  126,  127,  128,  129,  130,  131,  132,  133,  134,
  135,  136,  137,  138,  139,  140,  141,  142,  143,  144,  145,  146,
  147,  148,  149,  150,  151,  152,  153,  154,  155,  156,  157,  158,
  159,  160,    4,    8,   11,   14,   17,   20,   23,  259,  262,  265,
  268,  271,  274,  277,  441,  459,  462,  465,  469,  472,  475,  478,
  481,  484,  487,  490,  493,  496,  499,  502,  505,  508,  511,  514,
  161,  162,  163,  164,  165,  166,  167,  168,  169,  170,  171,  172,
  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,  183,  184,
  185,  186,  187,  188,  189,  190,  191,  192,  193,  194,  195,  196,
  197,  198,  199,  200,  201,  202,  203,  204,  205,  206,  207,  208,
  209,  210,  211,  212,  213,  214,  215,  216,  217,  218,  219,  220,
  221,  222,  223,  224,  225,  226,  227,  228,  229,  230,  231,  232,
  233,  234,  235,  236,  237,  238,  239,  240,  241,  242,  243,  244,
  245,  246,  247,  248,  249,  250,  251,  252,  253,  254,  255,  256,
  257,  258,  280,  281,  282,  283,  284,  285,  286,  287,  288,  289,
  290,  291,  292,  293,  294,  295,  296,  297,  298,  299,  300,  301,
  302,  303,  304,  305,  306,  307,  308,  309,  310,  311,  312,  313,
  314,  315,  316,  317,  318,  319,  320,  321,  322,  323,  324,  325,
  326,  327,  328,  329,  330,  331,  332,  333,  334,  335,  336,  337,
  338,  339,  340,  341,  342,  343,  344,  345,  346,  347,  348,  349,
  350,  351,  352,  353,  354,  355,  356,  357,  358,  359,  360,  361,
  362,  363,  364,  365,  366,  367,  368,  369,  370,  371,  372,  373,
  374,  375,  376,  377,  378,  379,  380,  381,  382,  383,  384,  385,
  386,  387,  388,  389,  390,  391,  392,  393,  394,  395,  396,  397,
  398,  399,  400,  401,  402,  403,  404,  405,  406,  407,  408,  409,
  410,  411,  412,  413,  414,  415,  416,  417,  418,  419,  420,  421,
  422,  423,  424,  425,  426,  427,  428,  429,  430,  431,  432,  433,
  434,  435,  436,  437,  438,  439,  440,  442,  443,  444,  445,  446,
  447,  448,  449,  450,  451,  452,  453,  454,  455,  456,  467]


#weight file
# file = open('weights/2259.pckl', 'rb')
# metadata = pickle.load(file)
# file.close()


#CNN
dataSet = 1458
training_data, validation_data, test_data, training_outputs = network3.load_data_shared(filename = str(dataSet)+'PartialHalf', perm = p)
mini_batch_size = 300
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 103, 5), 
                  filter_shape=(7, 1, 3, 3), 
                  poolsize=(1, 1)),
        ConvPoolLayer(image_shape=(mini_batch_size, 7, 101, 3), 
                  filter_shape=(7, 7, 3, 3), 
                  poolsize=(1, 1)),
        FullyConnectedLayer(n_in=7 * 99 * 1, n_out=600),
        FullyConnectedLayer(n_in=600, n_out=500),
        SoftmaxLayer(n_in=500, n_out=2)], mini_batch_size)
net.SGD(training_data, 1200000000, mini_batch_size, .5, 
            validation_data, test_data,training_outputs, weight_file = str(dataSet)+'CNN', variable_updates = False, log_file = str(dataSet)+'CNN',lmbda = .0001)


#MLP
# dataSet = 1458
training_data, validation_data, test_data, training_outputs = network3.load_data_shared(filename = str(dataSet)+'PartialHalf', perm = p)
mini_batch_size = 300
net = Network([
		FullyConnectedLayer(n_in=515, n_out=400),
    FullyConnectedLayer(n_in=400, n_out=400),
    SoftmaxLayer(n_in=400, n_out=2)], mini_batch_size)
net.SGD(training_data, 10000000, mini_batch_size, .1, 
            validation_data, test_data, training_outputs, weight_file = str(dataSet)+'MLP', variable_updates = False, log_file = str(dataSet)+'MLP', lmbda = .0001)


