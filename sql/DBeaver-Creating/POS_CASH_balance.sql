CREATE TABLE public.pos_cash_balance (
	sk_id_prev int4 NULL,
	sk_id_curr int4 NULL,
	months_balance int4 NULL,
	cnt_instalment numeric NULL,
	cnt_instalment_future numeric NULL,
	name_contract_status varchar(40) NULL,
	sk_dpd int4 NULL,
	sk_dpd_def int4 NULL
);