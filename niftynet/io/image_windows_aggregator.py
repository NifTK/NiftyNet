class ImageWindowsAggregator(object):

    def __init__(self, image_reader=None):
        self.reader = image_reader

    def decode_batch(self, batch_data):
        self.reader.output_list[0]['image'].shape
        import pdb; pdb.set_trace()
        return