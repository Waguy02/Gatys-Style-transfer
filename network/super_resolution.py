from ISR.models import RDN
class ISRModel:
    def __init__(self):
        self.model = RDN(weights='psnr-small')

    def process(self,image):
        return self.model.predict(image)*255