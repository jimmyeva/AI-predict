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
   
   Update_time          timestamp comment '更新写入时间',
   Goods_name       longtext comment '当前组合商品名称',
   T_profit              bigint comment '总毛利',
   Profit_Rate		longtext comment '每款商品对应的毛利率',
   Origin_sales			bigint comment '每款商品对应的原价',
   Profit			longtext comment '商品毛利',
   Discount_rate		bigint comment '商品毛利率',
   Event_price 			bigint comment '活动价',
   Rank					bigint comment '毛利排名',
   T_price				bigint comment '总订单金额',
   T_profit_rate		bigint comment '总毛利率',
   Platform_code      longtext comment '平台名称'
   
);

alter table NR_profit comment 'NR_profit';

