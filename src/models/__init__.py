"""模型包导出入口。

外部代码可以通过 `from src.models import ImprovedBayesianPVNet` 直接导入主模型。
"""

from src.models.improved_bnn import ImprovedBayesianPVNet

__all__ = ["ImprovedBayesianPVNet"]
