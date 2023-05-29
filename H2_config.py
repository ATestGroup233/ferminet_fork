
from ferminet import base_config
from ferminet.utils import system

def get_config():
    cfg = base_config.default()
    cfg.system.electrons = (1, 1)
    cfg.system.molecule = [system.Atom('H',(0, 0, -1)), system.Atom('H',(0, 0, 1))]
    
    cfg.batch_size = 256
    cfg.pretrain.iterations = 100 
    
    return cfg
