data=(admin_dis part_admin_dis)

for d in ${data[@]}; do
    args=(
        --preprocess_type ${d}
    )
    echo `python preprocess_data.py ${args[@]}`
done