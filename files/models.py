from django.db import models

# Create your models here.
class materials(models.Model):
    title = models.CharField(max_length = 100)
    files = models.FileField(upload_to='')

    def __str__(self):
        return self.title