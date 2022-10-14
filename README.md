# trail_cam_classifier

Start by runnign the jupyter notebook which will download day and night trail cam images of deer, elk and moose. Then go through the process of cleaning the data as some of the images are mislabeled. After this we train a better model that achieves 80% accuracy. This could probably be improved with better data aurmentation. When you are satidfied with your model accuracy export it to a pkl file. In the app.py (which is designed for a hugging face spaces gradio) you build your gadio app with the pickled model and some trail cam photo examples.
