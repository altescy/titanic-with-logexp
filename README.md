Titanic with logexp
===

```
$ logexp init -e titanic
$ logexp run -e [ EXP_ID ] -w sklearn_trainer -p configs/sklearn_trainer/lgbm.jsonnet
$ kaggle competition submit \
    -c titanic \
    -m "your message" \
    -f `logexp show -r [ RUN_ID ] | jq -r .storage.rootdir`/submit.csv
```
