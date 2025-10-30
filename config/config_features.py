from dataclasses import dataclass
from typing import List


@dataclass
class FeatureConfig:
    node_features: List[str]
    edge_features: List[str]
    redundant_columns: List[str] = None
    
    def __post_init__(self):
        if self.redundant_columns is None:
            self.redundant_columns = []


FEATURE_CONFIGS = {
    "bigbank_minimal": FeatureConfig(
        node_features=['weighted_out', 'out_degree', 'weighted_in', 'in_degree'],
        edge_features=['num_transactions', 'weight', 'num_payment_formats',
                      'num_payment_currencies', 'velocity', 'ratio', 'cov', 'gini'],
        redundant_columns=['amount_paid', 'payment_currency']
    ),
    
    "bigbank_network": FeatureConfig(
        node_features=['deg_centrality', 'betweenness', 'weighted_out', 'out_degree', 
                      'weighted_in', 'community', 'pagerank', 'in_degree'],
        edge_features=['num_transactions', 'weight', 'num_payment_formats',
                      'num_payment_currencies', 'velocity', 'ratio', 'cov', 'gini'],
        redundant_columns=['amount_paid', 'payment_currency']
    ),
    
    "bigbank_temporal": FeatureConfig(
        node_features=['weighted_out', 'out_degree', 'weighted_in', 'in_degree'],
        edge_features=['num_transactions', 'weight', 'velocity', 
                      'time_concentration', 'burst_score'],
        redundant_columns=['amount_paid', 'payment_currency']
    ),
}
