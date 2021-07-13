# License Injection Notes for AppVeyor CI

We use the recommended AppVeyor pattern to embed a key with a few months of lifetime into the repository so unit tests can run.

A few steps are taken to perform this.

In preparation, read this:
[https://www.appveyor.com/docs/how-to/secure-files/]


1. Download the secure-file tool to the workstation that will update the license (use this sub-folder, since it is already covered by .gitignore)
2. Create a working pair of files (use the .example files, and update the license) to encrypt
3. If you are not sure what the previously used secret was, generate a new secret against the AppVeyor project here: [https://ci.appveyor.com/tools/encrypt]
4. Encrypt the files, for example like this:

```
.\secure-file -encrypt ..\common\cinecoder_license_string.cs -secret somerandomsecret -out ..\common\cinecoder_license_string.cs.enc
```

5. Update the appveyor.yml file with the new salt values (and change the secured secret if changed) - DO NOT WRITE THE PLAINTEXT SECRET INTO THE YML
6. Commit the changes (the plaintext .cs and .h files from step 2 should be in .gitignore already)
7. Check the GitHub / AppVeyor build ran and turned green.