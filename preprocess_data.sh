data=(company_admin comp_admin_dis abs_admin_dis plain_admin_dis part_admin_dis)

for d in ${data[@]}; do
    args=(
        --preprocess_type ${d}
    )
    echo `python preprocess_data.py ${args[@]}`
done