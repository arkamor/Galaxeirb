%% init
clear all;
close all;

%% vars

var = "coucou";

%% work

disp(var);
M = dlmread('dubinski.tab',' ');
A = M(:,1);
B = M(:,2);
C = M(:,3);
D = M(:,4);
E = M(:,5);
F = M(:,6);
G = M(:,7);

f_mas = fopen('out/mas','wt');
f_pox = fopen('out/pox','wt');
f_poy = fopen('out/poy','wt');
f_poz = fopen('out/poz','wt');
f_vex = fopen('out/vex','wt');
f_vey = fopen('out/vey','wt');
f_vez = fopen('out/vez','wt');

for ii = 1:size(A,1)
    fprintf(f_mas,'%g\t',A(ii,:));
    fprintf(f_mas,'\n');
    
    fprintf(f_pox,'%g\t',B(ii,:));
    fprintf(f_pox,'\n');
    fprintf(f_poy,'%g\t',C(ii,:));
    fprintf(f_poy,'\n');
    fprintf(f_poz,'%g\t',D(ii,:));
    fprintf(f_poz,'\n');
    
    fprintf(f_vex,'%g\t',E(ii,:));
    fprintf(f_vex,'\n');
    fprintf(f_vey,'%g\t',F(ii,:));
    fprintf(f_vey,'\n');
    fprintf(f_vez,'%g\t',G(ii,:));
    fprintf(f_vez,'\n');
end


fclose(f_mas)
fclose(f_pox)
fclose(f_poy)
fclose(f_poz)
fclose(f_vex)
fclose(f_vey)
fclose(f_vez)

