data=(admin_dis part_admin_dis comp_admin_dis plain_admin_dis abs_admin_dis)

for d in ${data[@]}; do
    args=(
        --preprocess_type ${d}
    )
    echo `python main.py ${args[@]}`
done