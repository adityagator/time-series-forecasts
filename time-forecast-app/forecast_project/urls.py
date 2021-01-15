from django.contrib import admin
from django.urls import path
from pages.views import home_view
from pages.views import about_view
from pages.views import help_view
from pages.views import contact_view
from data.views import input_create_view
from data.views import dashboard_view
from data.views import covid_view
from data.views import demo_view
from processing.views import processing_view
from django.conf import settings
from django.conf.urls.static import static
from users import views as user_views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib.auth import views as auth_views

urlpatterns = [
    # page paths
    path('', home_view, name='home'),
    path('about/', about_view, name='about'),
    path('contact/', contact_view, name='contact'),
    path('help/', help_view, name='help'),
    path('create/', input_create_view, name='create'),
    path('process/<int:id>', processing_view),
    path('dashboard/<int:id>', dashboard_view),
    # user paths
    path('register/', user_views.register_view, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='logout.html'), name='logout'),
    path('password-reset/', auth_views.PasswordResetView.as_view(template_name='password_reset.html'), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='password_reset_done.html'), name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(
             template_name='password_reset_confirm.html'
         ),
         name='password_reset_confirm'),
    path('password-reset-complete/',
         auth_views.PasswordResetCompleteView.as_view(
             template_name='password_reset_complete.html'
         ),
         name='password_reset_complete'),
    # covid
    path('demo/', demo_view, name='demo'),
    path('covid/', covid_view, name='covid'),
    path('admin/', admin.site.urls),
]
urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
urlpatterns += staticfiles_urlpatterns()