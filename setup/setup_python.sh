cd setup
if [ ! -d py_env ]; then
   if [ "${ARCH}" = "i686" ]; then
     python3.8 -m venv py_env
   else
     python3.8 -m venv py_env
   fi

   source py_env/bin/activate
   pip install --upgrade pip

   pip install pyyaml
   pip install pandas
   pip install scipy
   pip install datetime
   pip install matplotlib
   pip install shutils
   pip install psutil
   pip install torch
   pip install torchvision
   pip install sklearn
   pip install scikit-learn==0.22.2.post1

else
   source py_env/bin/activate
fi
cd -