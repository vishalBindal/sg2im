if [[ $PYTHONPATH =~ $PWD ]]
then
    echo ""
else
    export PYTHONPATH=$PWD:$PYTHONPATH
fi

python3 scripts/run_model.py \
  --checkpoint "checkpoint_with_model.pt" \
  --scene_graphs test_image.json \
  --output_dir outputs