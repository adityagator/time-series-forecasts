# Generated by Django 3.0.5 on 2020-04-08 23:48

from django.db import migrations
import jsonfield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0010_auto_20200407_0658'),
    ]

    operations = [
        migrations.AddField(
            model_name='outputdata',
            name='int_cluster',
            field=jsonfield.fields.JSONField(default={'default': 'default'}),
        ),
        migrations.AddField(
            model_name='outputdata',
            name='volume_cluster',
            field=jsonfield.fields.JSONField(default={'default': 'default'}),
        ),
    ]
