"""forecast_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from pages.views import home_view
from pages.views import about_view
from pages.views import help_view
from pages.views import contact_view
from data.views import input_create_view
from data.views import output_detail_view
from data.views import dashboard_view, dashboard_calculate
from processing.views import processing_view
from django.conf import settings
from django.conf.urls.static import static
from users import views as user_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib.auth import views as auth_views
# from pages.templates import home.html as landing_page

urlpatterns = [
    path('', home_view, name='home'),
    path('about/', about_view, name='about'),
    path('contact/', contact_view, name='contact'),
    path('help/', help_view, name='help'),
    path('create/', input_create_view, name='create'),
    path('process/<int:id>', processing_view),
    path('output/<int:id>', output_detail_view),
    path('dashboard/<int:id>', dashboard_view),
    path('dashcalculate/<int:id>/', dashboard_calculate, name='dash-calc'),
    path('register/', user_views.register_view, name='register'),
    path('profile/', user_views.profile_view, name='profile'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='home.html'), name='logout'),
    # path('home/<str:username>', user_views.user_home_view, name='user_home'),
    path('home/', user_views.user_home_view, name='user_home'),
    path('admin/', admin.site.urls),
]
urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
# urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += staticfiles_urlpatterns()