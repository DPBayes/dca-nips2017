#!/bin/bash

if [ ! -f "secret.txt" ]; then echo "Please create secret.txt with a single line containing the desired private key password."; exit 1; fi
# Store passwords in a separate file
pass=$( cat secret.txt )

res=$PWD
ks=$res/keystore.jks

genkey(){
  str="PROBIC-${1}\nDepartment of Computer Science\nUniversity of Helsinki\nHelsinki\nUusimaa\nFI\nyes"
  echo -e $str | keytool -genkey -alias "probic-${1}" -keyalg RSA -keystore $ks -keysize 4096 -storepass $pass -keypass $pass -validity 360
}

