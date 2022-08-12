import torch
def xyz2rgb(xyz):
    # RGB = RGB / 255
    # v1 = RGB / 12.92
    # v2 = torch.pow((RGB + 0.055) / 1.055, 2.4)
    # RGB = torch.where(RGB <= 0.04045, v1, v2)
    R = 2.4935 * xyz[:, 0, :, :] - 0.9315 * xyz[:, 1, :, :] - 0.4027 * xyz[:, 2, :, :]
    G =-0.8296 * xyz[:, 0, :, :] + 1.7629 * xyz[:, 1, :, :] + 0.0236 * xyz[:, 2, :, :]
    B = 0.0358 * xyz[:, 0, :, :] - 0.0762 * xyz[:, 1, :, :] + 0.9571 * xyz[:, 2, :, :]
    R = torch.unsqueeze(R, 1)
    G = torch.unsqueeze(G, 1)
    B = torch.unsqueeze(B, 1)
    RGB = torch.cat([R,G,B], dim=1)
    # v1 = RGB * 12.92
    # v2 = torch.pow(RGB*1.055, 1/2.4)-0.055
    RGB = torch.where(RGB <= 0.00304, RGB * 12.92, (torch.pow(RGB*1.055, 1/2.4)-0.055))
    return RGB

def rgb2xyz(RGB):
    X = 0.4866*RGB[:,0,:,:] + 0.2657*RGB[:,1,:,:] + 0.1982*RGB[:,2,:,:]
    Y = 0.2290*RGB[:,0,:,:] + 0.6917*RGB[:,1,:,:] + 0.0793*RGB[:,2,:,:]
    Z = 0.0000*RGB[:,0,:,:] + 0.0451*RGB[:,1,:,:] + 1.0437*RGB[:,2,:,:]
    X = torch.unsqueeze(X,1)
    Y = torch.unsqueeze(Y,1)
    Z = torch.unsqueeze(Z,1)
    Image = torch.cat([X,Y,Z],dim=1)
    return Image

def rgb2xyzmath(RGB):
    RGB = RGB / 255
    v1 = RGB / 12.92
    v2 = torch.pow((RGB + 0.055) / 1.055, 2.4)
    RGB = torch.where(RGB <= 0.04045, v1, v2)
    X = 0.4866 * RGB[:, 0, :, :] + 0.2657 * RGB[:, 1, :, :] + 0.1982 * RGB[:, 2, :, :]
    Y = 0.2290 * RGB[:, 0, :, :] + 0.6917 * RGB[:, 1, :, :] + 0.0793 * RGB[:, 2, :, :]
    Z = 0.0000 * RGB[:, 0, :, :] + 0.0451 * RGB[:, 1, :, :] + 1.0437 * RGB[:, 2, :, :]
    X = torch.unsqueeze(X, 1)
    Y = torch.unsqueeze(Y, 1)
    Z = torch.unsqueeze(Z, 1)
    Image = torch.cat([X, Y, Z], dim=1)

    return Image

def rgb2yiq(RGB):
    Y = 0.299*RGB[:,0,:,:]+0.587*RGB[:,1,:,:]+0.114*RGB[:,2,:,:]
    I = 0.596*RGB[:,0,:,:]-0.275*RGB[:,1,:,:]-0.321*RGB[:,2,:,:]
    Q = 0.212*RGB[:,0,:,:]-0.523*RGB[:,1,:,:]+0.311*RGB[:,2,:,:]
    Y = torch.unsqueeze(Y,1)
    I = torch.unsqueeze(I,1)
    Q = torch.unsqueeze(Q,1)
    Image = torch.cat([Y,I,Q],dim=1)
    return Image