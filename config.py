from pathlib import Path

__all__ = ['project_path', 'dataset_config']

project_path = Path(__file__).parent


dataset_config = {
    'elliptic':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':64,
            'lr_f': 1e-4,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
    'yelp':
        {
            'K': 2,
            'M': 5,
            'm': 3,
            'hidden_channels': 64,
            'out_channels': 64,
            'lr_f': 5e-3,
            'lr': 5e-3,
            'weight_decay': 5e-4,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
    'weibo':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':64,
            'lr_f': 1e-3,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'beta': 0.2,
            'epochs': 1000,
            'patience': 200
        },
    'quest':
        {
            'K': 2,
            'M': 5,
            'm': 3,
            'hidden_channels': 64,
            'out_channels': 64,
            'lr_f': 5e-3,
            'lr': 5e-3,
            'weight_decay': 5e-4,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
    'Amazon':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':64,
            'lr_f': 5e-4,
            'lr': 5e-4,
            'weight_decay': 5e-4,
            'beta': 0.5,
            'epochs': 100,
            'patience': 200
        },
    'ACM':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':64,
            'lr_f': 1e-4,
            'lr': 1e-2,
            'weight_decay': 1e-4,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
        'tfinance':
        {
            'K': 1,
            'M': 1,
            'm': 4,
            'hidden_channels':64,
            'lr_f': 1e-4,
            'lr': 1e-2,
            'weight_decay': 1e-4,
            'beta': 0.5,
            'epochs': 1000,
            'patience': 200
        },
}