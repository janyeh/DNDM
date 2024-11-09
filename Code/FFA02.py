import torch.nn as nn
import torch

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)

def safe_clamp_tuple(tuple_tensor, name="", min=-1.0, max=1.0):
    """
    Safely clamp tuple of tensors, handling None values and providing warnings
    Returns tuple of clamped tensors
    """
    if not isinstance(tuple_tensor, tuple):
        return tuple_tensor
    result = []
    for i, tensor in enumerate(tuple_tensor):
        if tensor is not None:
            if not torch.isfinite(tensor).all():
                print(f"Warning: Non-finite values detected in {name}[{i}]. Clamping values.")
            result.append(torch.clamp(tensor, min=min, max=max))
        else:
            result.append(None)
    return tuple(result)

def safe_clamp(tensor, name="", min=-1.0, max=1.0): 
    return safe_clamp_tuple((tensor,), name)[0]

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class ffa(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(ffa, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, 1,blocks=blocks)
        self.g2= Group(conv, self.dim, 3,blocks=blocks)
        self.g3= Group(conv, self.dim, 5,blocks=blocks)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1, meta):
        # JanYeh DEBUG BEGIN
        # JanYeh: Clamp meta before using it in channel attention
        meta = torch.clamp(meta, min=-1.0, max=1.0)
        
        x = safe_clamp(self.pre(x1), "pre")

        # JanYeh: Check for NaN or Inf after pre-processing
        
        res1=safe_clamp(self.g1(x), "g1")
        res2=safe_clamp(self.g2(res1), "g2")
        res3=safe_clamp(self.g3(res2), "g3")
        # JanYeh: Clamp and Check for NaN or Inf after group layers
        
        w=safe_clamp(self.ca(meta), "ca")

        # JanYeh: clamp the weights to prevent exploding gradient
        # Print the shapes of the tensors for debugging
        # print(f"res1 shape: {res1.shape}")
        # print(f"res2 shape: {res2.shape}")
        # print(f"res3 shape: {res3.shape}")
        # print(f"w shape before view: {w.shape}")

        # Reshape and apply weights to residuals
        w=safe_clamp(w.view(-1,self.gps,self.dim)[:,:,:,None,None], "view")
        # JanYeh: Check for NaNs or Infs in w and handle them

        out=safe_clamp(w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3, "out")
        # JanYeh: Check for NaNs or Infs in out and handle them

        # Memory usage after channel attention
        # print(f"Memory allocated after Channel Attention: {torch.cuda.memory_allocated()} bytes")
        # print(f"Max memory allocated so far: {torch.cuda.max_memory_allocated()} bytes")

        out=safe_clamp(self.palayer(out), "palayer")
        x=safe_clamp(self.post(out), "post")

        # Memory usage after final layer
        # print(f"Memory allocated after PALayer and Post Layer: {torch.cuda.memory_allocated()} bytes")
        # print(f"Max memory allocated so far: {torch.cuda.max_memory_allocated()} bytes")
        # JanYeh DEBUG END

        return x #+ x1

class ffa1(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(ffa1, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1=self.g1(x)
        res2=self.g2(res1)
        res3=self.g3(res2)
        w=self.ca(torch.cat([res1,res2,res3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
        out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3
        out=self.palayer(out)
        x=self.post(out)
        return x + x1
class ffa2(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(ffa2, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1=self.g1(x)
        res2=self.g2(res1)
        res3=self.g3(res2)
        w=self.ca(torch.cat([res1,res2,res3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
        out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3
        out=self.palayer(out)
        x=self.post(out)
        return x + x1
class ffa3(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(ffa3, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)

        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1=self.g1(x)
        res2=self.g2(res1)
        res3=self.g3(res2)
        w=self.ca(torch.cat([res1,res2,res3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
        out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3
        out=self.palayer(out)
        x=self.post(out)
        return x + x1
if __name__ == "__main__":
    net=ffa(gps=3,blocks=19)
    print(net)