data=(abs_company_admin comp_company_admin plain_company_admin part_company_admin company_admin abs_admin_dis comp_admin_dis plain_admin_dis part_admin_dis admin_dis)

for d in ${data[@]}; do
    args=(
        --preprocess_type ${d}
    )
    echo `python preprocess_data.py ${args[@]}`
done