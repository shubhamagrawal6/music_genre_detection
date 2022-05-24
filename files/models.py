from django.db import models
import os
from django.db import models
from datetime import datetime

# Create your models here.

class materials(models.Model):
        
    def rename(path):
        def wrapper(instance, filename):
            d=datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{d}.wav"
            return os.path.join(path,filename)
        return wrapper

    title = models.CharField(max_length = 100)
    files = models.FileField(upload_to=rename(''))

    def str(self):
        return self.title