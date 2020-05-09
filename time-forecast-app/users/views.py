from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .forms import UserRegisterForm
from django.contrib.auth.decorators import login_required
def register_view(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Resgistered as {username}')
            # link_with_id = '/home/' + str(username)
            return redirect('user_home')
    else:
        form = UserRegisterForm()
    return render(request, 'register.html', {'form': form})

def user_home_view(request, username):
    context = {
        'username' : username
    }
    return render(request, 'user_home.html', context)
@login_required
def user_home_view(request):
    context = {
        # 'username' : username
    }
    return render(request, 'user_home.html', context)

@login_required
def profile_view(request):
    return render(request, 'profile.html', {})