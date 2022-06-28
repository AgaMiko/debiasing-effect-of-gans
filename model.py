from torch import nn


class Net(nn.Module):
    def __init__(self, arch, return_feats=False):
        super(Net, self).__init__()
        self.arch = arch
        self.return_feats = return_feats
        if 'EfficientNet' in str(arch.__class__):   
            self.arch._fc = nn.Linear(in_features=self.arch._fc.in_features, out_features=500, bias=True)
        else:   
            self.arch.fc = nn.Linear(in_features=arch.fc.in_features, out_features=500, bias=True)
            
        self.ouput = nn.Linear(500, 1)
        
    def forward(self, images):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x = images
        features = self.arch(x)
        output = self.ouput(features)
        if self.return_feats:
            return features
        return output