import torch
import torch.optim as optim

from src.models import create_model
from src.proden_loss import proden
from src.clpl_loss import CLPLSquaredHingeLoss
from src.mcl_losses import MCL_LOG, MCL_MAE, MCL_EXP
from src.pico.model import PiCOModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.solar.utils_loss import partial_loss as solar_partial_loss
from src.comco.model import ComCoModel
from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss

def setup_cour(args, train_config):
    """Initializes model, loss, and optimizer for Cour 2011 CLPL (squared-hinge)."""
    model = create_model(train_config['num_classes'])
    loss = CLPLSquaredHingeLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss, optimizer

def setup_proden(args, train_config):
    """Initializes model, loss, and optimizer for PRODEN."""
    model = create_model(train_config['num_classes'])
    loss = proden()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss, optimizer

def setup_mcl(args, train_config, loss_type='log'):
    """Initializes model, loss, and optimizer for MCL."""
    model = create_model(train_config['num_classes'])
    if loss_type == 'log':
        loss = MCL_LOG(num_classes=train_config['num_classes'])
    elif loss_type == 'mae':
        loss = MCL_MAE(num_classes=train_config['num_classes'])
    elif loss_type == 'exp':
        loss = MCL_EXP(num_classes=train_config['num_classes'])
    else:
        raise ValueError(f"Unknown MCL loss type: {loss_type}")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss, optimizer

def setup_scl(args, train_config):
    """Initializes model, loss, and optimizer for SCL-NL (Chou et al. 2020)."""
    from src.scl_loss import SCL_NL
    model     = create_model(train_config['num_classes'])
    loss      = SCL_NL()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss, optimizer


def setup_comco(args, train_config, comco_config, device):
    """Initializes model, losses, and optimizer for ComCo."""
    from src.comco.model import ComCoModel
    from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss

    comco_args = {
        'num_class':   train_config['num_classes'],
        'epochs':      args.epochs,
        'low_dim':     comco_config['low_dim'],
        'moco_queue':  comco_config['moco_queue'],
        'moco_m':      comco_config['moco_m'],
        'loss_weight': comco_config['loss_weight'],
        'warmup_neg':  comco_config['warmup_neg'],
        'warmup_pos':  comco_config['warmup_pos'],
    }
    model     = ComCoModel(comco_args).to(device)
    cls_loss  = ComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(
        temperature=comco_config['temperature'],
        top_k=comco_config['top_k'],
    )
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    return model, (cls_loss, cont_loss), optimizer, comco_args


def setup_pico(args, train_config, pico_config, pico_train_dataset, device):
    """Initializes model, losses, and optimizer for PiCO."""
    pico_args = {
        'num_class': train_config['num_classes'], 'epochs': args.epochs, 'low_dim': pico_config['low_dim'],
        'moco_queue': pico_config['moco_queue'], 'moco_m': pico_config['moco_m'], 'proto_m': pico_config['proto_m'],
        'prot_start': pico_config['prot_start'], 'loss_weight': pico_config['loss_weight'],
        'conf_ema_range': pico_config['conf_ema_range']
    }
    model = PiCOModel(pico_args).to(device)
    
    initial_confidence = torch.ones(len(pico_train_dataset), pico_args['num_class']) / pico_args['num_class']
    cls_loss = PartialLoss(initial_confidence.to(device))
    cont_loss = SupConLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    return model, (cls_loss, cont_loss), optimizer, pico_args

def setup_comco(args, train_config, comco_config, device):
    """Initializes model, losses, and optimizer for ComCo."""
    comco_args = {
        'num_class': train_config['num_classes'],
        'epochs': args.epochs,
        'low_dim': comco_config['low_dim'],
        'moco_queue': comco_config['moco_queue'],
        'moco_m': comco_config['moco_m'],
        'loss_weight': comco_config['loss_weight'],
        'temperature': comco_config['temperature'],
        'top_k': comco_config['top_k'],
        'warmup_neg': comco_config['warmup_neg'],
        'warmup_pos': comco_config['warmup_pos'],
    }
    model = ComCoModel(comco_args).to(device)
    cls_loss = ComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'], top_k=comco_args['top_k'])
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return model, (cls_loss, cont_loss), optimizer, comco_args


def setup_wu(args, train_config):
    """Initializes model, loss, and optimizer for Wu et al. proper PLL."""
    from src.wu_loss import WuPLLLoss
    model     = create_model(train_config['num_classes'])
    loss      = WuPLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    return model, loss, optimizer


def setup_pico_mclloss(args, train_config, pico_config, device):
    """Initializes PiCO with MCL-LOG partial label cls loss (no EMA confidence matrix)."""
    from src.pico.mcl_cls_loss import PiCOMCLLoss
    pico_args = {
        'num_class':      train_config['num_classes'],
        'epochs':         args.epochs,
        'low_dim':        pico_config['low_dim'],
        'moco_queue':     pico_config['moco_queue'],
        'moco_m':         pico_config['moco_m'],
        'proto_m':        pico_config['proto_m'],
        'prot_start':     pico_config['prot_start'],
        'loss_weight':    pico_config['loss_weight'],
        'conf_ema_range': pico_config['conf_ema_range'],
    }
    model     = PiCOModel(pico_args).to(device)
    cls_loss  = PiCOMCLLoss()
    cont_loss = SupConLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    return model, (cls_loss, cont_loss), optimizer, pico_args


def setup_solar(args, train_config, solar_config, solar_train_dataset, device):
    """Initializes model, loss, and optimizer for SoLar."""
    solar_args = {
        'num_class': train_config['num_classes'], 'epochs': args.epochs, 'warmup_epoch': solar_config['warmup_epoch'],
        'rho_range': solar_config['rho_range'], 'lamd': solar_config['lamd'], 'eta': solar_config['eta'],
        'tau': solar_config['tau'], 'est_epochs': solar_config['est_epochs'], 'gamma1': solar_config['gamma1'],
        'gamma2': solar_config['gamma2']
    }
    model = create_model(train_config['num_classes']).to(device)
    
    num_classes = train_config['num_classes']
    solar_given_label_matrix = torch.zeros(len(solar_train_dataset), num_classes)
    for i, p_label in enumerate(solar_train_dataset.given_label_matrix_sparse):
        solar_given_label_matrix[i, p_label] = 1.0
        
    loss_fn = solar_partial_loss(solar_given_label_matrix, device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    queue = torch.zeros(64 * args.batch_size, train_config['num_classes']).to(device)
    
    return model, loss_fn, optimizer, solar_args, queue
