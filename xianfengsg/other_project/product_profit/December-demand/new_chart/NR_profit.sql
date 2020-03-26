/*==============================================================*/
/* DBMS name:      MySQL 5.0                                    */
/* Created on:     2019/12/15 10:08:25                          */
/*==============================================================*/


drop table if exists NR_profit;

/*==============================================================*/
/* Table: NR_profit                                          */
/*==============================================================*/
create table NR_profit
(
   
   Update_time          timestamp comment '����д��ʱ��',
   Goods_name       longtext comment '��ǰ�����Ʒ����',
   T_profit              bigint comment '��ë��',
   Profit_Rate		longtext comment 'ÿ����Ʒ��Ӧ��ë����',
   Origin_sales			bigint comment 'ÿ����Ʒ��Ӧ��ԭ��',
   Profit			longtext comment '��Ʒë��',
   Discount_rate		bigint comment '��Ʒë����',
   Event_price 			bigint comment '���',
   Rank					bigint comment 'ë������',
   T_price				bigint comment '�ܶ������',
   T_profit_rate		bigint comment '��ë����',
   Platform_code      longtext comment 'ƽ̨����'
   
);

alter table NR_profit comment 'NR_profit';

