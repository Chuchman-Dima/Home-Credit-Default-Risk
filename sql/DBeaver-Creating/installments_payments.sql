CREATE TABLE public.installments_payments (
	sk_id_prev int4 NULL,
	sk_id_curr int4 NULL,
	num_stalment_version numeric NULL,
	num_instalment_number int4 NULL,
	days_instalment numeric NULL,
	days_entry_payment numeric NULL,
	amt_instalment numeric(15, 2) NULL,
	amt_payment numeric(15, 2) NULL
);