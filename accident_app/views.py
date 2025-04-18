
from django.shortcuts import render
from .forms import VideoUploadForm
from django.conf import settings
import os
from .model_loader import predict_from_video

def predict_view(request):
    result = None
    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = form.cleaned_data['video_file']
            save_path = os.path.join(settings.MEDIA_ROOT, video_file.name)
            with open(save_path, 'wb+') as dest:
                for chunk in video_file.chunks():
                    dest.write(chunk)
            pred, prob = predict_from_video(save_path)
            result = {
                'prediction': 'Accident' if pred else 'No Accident',
                'confidence': f"{prob*100:.2f}%"
            }
    else:
        form = VideoUploadForm()
    return render(request, 'accident_app/predict.html', {'form': form, 'result': result})


def dashboard_view(request):
    # Dummy stats for example
    stats = {
        'accidents_today': 5,
        'safe_roads': 12,
        'last_prediction': 'Accident',
    }
    return render(request, 'accident_app/dashboard.html', {'stats': stats})
