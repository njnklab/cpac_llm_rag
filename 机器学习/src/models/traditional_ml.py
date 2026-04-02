"""
传统机器学习模型：SVM和Random Forest
用于FC矩阵分类（单中心/多中心数据）
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Dict, Any, Tuple


class SVMClassifier:
    """SVM分类器包装类"""
    
    def __init__(self, C: float = 0.01, kernel: str = 'linear', 
                 class_weight: str = 'balanced', random_state: int = 42):
        self.C = C
        self.kernel = kernel
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.selector = None
        self.k_best = 200
        
    def fit(self, X: np.ndarray, y: np.ndarray, k_best: int = 200):
        self.k_best = k_best
        self.selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = self.selector.fit_transform(X, y)
        X_scaled = self.scaler.fit_transform(X_selected)
        
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            class_weight=self.class_weight,
            probability=True,
            random_state=self.random_state
        )
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_selected = self.selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        return y_pred, y_prob


class RFClassifier:
    """Random Forest分类器包装类"""
    
    def __init__(self, n_estimators: int = 150, max_depth: int = 10,
                 class_weight: str = 'balanced', random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.selector = None
        self.k_best = 200
        
    def fit(self, X: np.ndarray, y: np.ndarray, k_best: int = 200):
        self.k_best = k_best
        self.selector = SelectKBest(score_func=f_classif, k=k_best)
        X_selected = self.selector.fit_transform(X, y)
        X_scaled = self.scaler.fit_transform(X_selected)
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features='sqrt',
            min_samples_leaf=3,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_selected = self.selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        return y_pred, y_prob


def get_classifiers() -> Dict[str, Any]:
    """获取所有分类器实例"""
    return {
        'Linear SVM': SVMClassifier(),
        'Random Forest': RFClassifier()
    }