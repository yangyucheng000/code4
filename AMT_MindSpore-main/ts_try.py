# import sys
# sys.path.insert(0, "path to your mindspore model define project")
# sys.path.insert(0, "path to your pytorch model define project")
import troubleshooter as ts
import numpy as np
from networks.AMT_S_Pytorch import Model as TorchNet
from networks.AMT_S_MindSpore import Model as MSNet

pt_net = TorchNet()
ms_net = MSNet()
diff_finder = ts.migrator.NetDifferenceFinder(
    pt_net=pt_net,
    ms_net=ms_net)
diff_finder.compare(auto_inputs=(((1, 12), np.float32), ))
