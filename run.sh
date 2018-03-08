echo "NUMBER 1"
time python ens.py 0
cp -r models ./night_save/mod0
echo "NUMBER 2"
time python ens.py 1
cp -r models ./night_save/mod1
echo "NUMBER 3"
time python ens.py 1
cp -r models ./night_save/mod2
echo "NUMBER 4"
time python ens.py 1
cp -r models ./night_save/mod3
echo "NUMBER 5"
time python ens.py 1
cp -r models ./night_save/mod4