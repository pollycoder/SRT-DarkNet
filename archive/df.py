import torch
from torch import nn

class DF(nn.Module):
    def __init__(self, num_classes=2, final_length=18):
        super(DF, self).__init__()
        
        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]
        pool_padding = 0
        length_after_extraction = final_length
        
        self.feature_extraction = nn.Sequential(
            # block1
            nn.Conv1d(in_channels=1,out_channels=filter_num[1],kernel_size=kernel_size[1],
                      stride=conv_stride_size[1],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0, inplace=True),
            nn.Conv1d(in_channels=filter_num[1],out_channels=filter_num[1],kernel_size=kernel_size[1],
                      stride=conv_stride_size[1],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[1]),
            nn.ELU(alpha=1.0, inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=pool_padding),
            nn.Dropout(p=0.1),
            
            # block2
            nn.Conv1d(in_channels=filter_num[1],out_channels=filter_num[2],kernel_size=kernel_size[2],
                      stride=conv_stride_size[2],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=filter_num[2],out_channels=filter_num[2],kernel_size=kernel_size[2],
                      stride=conv_stride_size[2],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=pool_padding),
            nn.Dropout(p=0.1),
            
            # block3
            nn.Conv1d(in_channels=filter_num[2],out_channels=filter_num[3],kernel_size=kernel_size[3],
                      stride=conv_stride_size[3],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=filter_num[3],out_channels=filter_num[3],kernel_size=kernel_size[3],
                      stride=conv_stride_size[3],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=pool_padding),
            nn.Dropout(p=0.1),
            
            # block4
            nn.Conv1d(in_channels=filter_num[3],out_channels=filter_num[4],kernel_size=kernel_size[4],
                      stride=conv_stride_size[4],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=filter_num[4],out_channels=filter_num[4],kernel_size=kernel_size[4],
                      stride=conv_stride_size[4],padding='same',bias=False),
            nn.BatchNorm1d(filter_num[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=pool_padding),
            nn.Dropout(p=0.1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=filter_num[4]*length_after_extraction,out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            
            nn.Linear(in_features=512, out_features=512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(in_features=512, out_features=2),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        #print("--after extraction:", x.shape)
        x = self.classifier(x)
        return x