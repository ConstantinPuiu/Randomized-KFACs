from vgg_model import vgg16_bn

def get_network(network, **kwargs):
    '''networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn

    }'''   
    networks = {'vgg16_bn': vgg16_bn}

    return networks[network](**kwargs)

