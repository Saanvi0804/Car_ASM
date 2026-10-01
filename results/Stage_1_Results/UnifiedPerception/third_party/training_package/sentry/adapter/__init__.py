"""Feature adapters that bridge backbone outputs to head inputs."""

from sentry.adapter.fpn_adapter import FPNChannelAdapter
from sentry.adapter.fpn_adapter_v2 import FPNSpatialAdapter
from sentry.adapter.feature_align_loss import FeatureAlignmentLoss, TeacherFeatureExtractor
from sentry.adapter.feature_align_loss_v2 import FeatureAlignmentLoss_v2
from sentry.adapter.yolo_neck_runner import YOLONeckRunner
