from niftynet.application.segmentation_application import SegmentationApplication

# For now, we just need different automatic logs since miss rate is not meaningful
class AutoencoderApplication(SegmentationApplication):
  def logs(self,train_dict,net_outputs):
    predictions=net_outputs
    labels = train_dict['Sampling/labels']
    return []
