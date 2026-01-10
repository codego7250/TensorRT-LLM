#you need to run twine and input the username and password
TWINE_USERNAME=oauth2accesstoken 
TWINE_PASSWORD=$(gcloud auth print-access-token --project public-py) 
echo $TWINE_USERNAME
echo $TWINE_PASSWORD
echo "twine upload --repository-url https://us-python.pkg.dev/fireworks-public/public-py/ ./dist-dir/tensorrt_llm-1.2.0rc13-cp312-cp312-linux_x86_64.whl"
twine upload --repository-url https://us-python.pkg.dev/fireworks-public/public-py/ ./dist-dir/tensorrt_llm-1.2.0rc13-cp312-cp312-linux_x86_64.whl 
