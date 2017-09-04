#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;


#define CENTRAL_CROP true   //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����

int main()
{
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������

	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����

	char filename[100];	
	char filename1[100];	
	
	for(int num=1; num<10; num++)
	{
		
		sprintf(filename,"F:/�о���/��Ƶ/train/%d.png",num);
		
		Mat src = imread(filename);//��ȡͼƬ

		vector<float> descriptors;//HOG����������

		hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
	
		//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
		if( 1 == num )
		{
			DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
			//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
			sampleFeatureMat = Mat::zeros(20, DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
			sampleLabelMat = Mat::zeros(20, 1, CV_32FC1);
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<float>(num,0) = 1;//���������Ϊ1������
	}

	//���ζ�ȡ������ͼƬ������HOG������
	for(int num=1; num<10 ; num++)
	{
		sprintf(filename1,"F:/�о���/��Ƶ/test/%d.png",num);
		Mat src = imread(filename1);//��ȡͼƬ
		//resize(src,img,Size(64,128));

		vector<float> descriptors;//HOG����������
		hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//cout<<"������ά����"<<descriptors.size()<<endl;

		for(int i=0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num+10,i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<float>(num+10,0) = -1;//���������Ϊ-1������
	}
	
	

	cout<<DescriptorDim;
	////���������HOG�������������ļ�
	ofstream fout("Feature.txt");
	for(int i=1; i<20; i++)
	{
		fout<<i<<endl;
		for(int j=1; j<DescriptorDim; j++)
			fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		fout<<endl;
	}

}


/*
vector<string> get(string filename)    
{    
	vector<string> img_path;//�����ļ�������       
	ifstream svm_data(filename.c_str());    
	string buf;    
	int len=0;    
	while( svm_data )//��ѵ�������ļ����ζ�ȡ����        
	{       

		if( getline( svm_data, buf ) )        
		{       
			//if( len%2==0)    
			img_path.push_back( buf );//ͼ��·��        
		}    
		//len++;    
	}    
	svm_data.close();//�ر��ļ�   
	return img_path;  
}  
int _tmain(int argc, _TCHAR* argv[])  
{  
	vector<string> img_path;//�����ļ�������     
	vector<int> img_catg;      
	int nLine = 0;      
	string buf;      
	locale loc = locale::global(locale(""));  
	ifstream svm_data( "pos.txt" );//ѵ������ͼƬ��·����д�����txt�ļ��У�ʹ��bat�������ļ����Եõ����txt�ļ�  
	locale::global(loc);  

	ofstream zhengyangben_txt;  
	float result;  


	unsigned long n;       
	while( svm_data )//��ѵ�������ļ����ζ�ȡ����      
	{      
		if( getline( svm_data, buf ) )      
		{      
			nLine ++;      
			if( nLine % 2 == 0 )//ע����������ͼƬȫ·����ż�����Ǳ�ǩ   
			{      
				img_catg.push_back( atoi( buf.c_str() ) );//atoi���ַ���ת�������ͣ���־(0,1��2��...��9)��ע����������Ҫ��������𣬷�������      
			}      
			else      
			{      
				img_path.push_back( buf );//ͼ��·��      
			}      

		}      
	}   
	int num=0;  
	svm_data.close();//�ر��ļ�      
	CvMat *data_mat, *res_mat;      
	int nImgNum = nLine / 2; //nImgNum������������ֻ���ı�������һ�룬��һ���Ǳ�ǩ       
	data_mat = cvCreateMat( nImgNum, 1764, CV_32FC1 );  //�ڶ���������������������������descriptors�Ĵ�С�����ģ�������descriptors.size()�õ����Ҷ��ڲ�ͬ��С������ѵ��ͼƬ�����ֵ�ǲ�ͬ��    
	cvSetZero( data_mat );      
	//���;���,�洢ÿ�����������ͱ�־      
	res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );      
	cvSetZero( res_mat );      
	IplImage* src;      
	IplImage* trainImg=cvCreateImage(cvSize(64,64),8,3);//��Ҫ������ͼƬ������Ĭ���趨ͼƬ��64*64��С���������涨����1764�����Ҫ����ͼƬ��С����������debug�鿴һ��descriptors�Ƕ��٣�Ȼ���趨��������      

	//����HOG����    
	for( string::size_type i = 0; i != img_path.size(); i++ )      
	{      
		src=cvLoadImage(img_path[i].c_str(),1);      
		if( src == NULL )      
		{      
			cout<<" can not load the image: "<<img_path[i].c_str()<<endl;      
			continue;      
		}      

		cvResize(src,trainImg);   

		HOGDescriptor *hog=new HOGDescriptor(cvSize(64,64),cvSize(16,16),cvSize(8,8),cvSize(8,8),9,1,-1,HOGDescriptor::L2Hys,0.200000,false);        
		vector<float>descriptors;//��Ž��       
		hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //Hog��������        

		cout<<"HOG dims: "<<descriptors.size()<<endl;          
		n=0;      
		for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)      
		{      
			cvmSet(data_mat,i,n,*iter);//�洢HOG����   
			n++;      
		}         


		if(img_catg[i]==-1)  
			img_catg[i]=2;  
		else  
			num++;  
		cvmSet( res_mat, i, 0, img_catg[i] );      
		// cout<<" �������: "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;      
	}      

	Mat p_data=Mat(data_mat);  
	Mat mean_=Mat(1,1764,CV_32FC1);  
	Mat mm = Mat(1,num,CV_32FC1);  
	mm=1;  
	gemm(mm,p_data,1,mean_,0,mean_);  
	mean_=mean_/num;  
	PCA pca;  
	pca(p_data,mean_,0,100);//ѡ����������  
	Mat mmm=pca.eigenvalues;  
	Mat test=p_data(Rect(0,0,1764,1));  
	Mat dd = Mat(mmm.size(),CV_32FC1);  
	pca.project(test,dd); //ͶӰ  
	Mat rec;  
	pca.backProject(dd,rec); //����ͶӰ����ԭ  
	printf("%d. diff = %g\n", 1, norm(test, rec, NORM_L2));  

	SVD svd;  
	svd(p_data);  
	Mat v=svd.vt;  

	Mat cc=v(Rect(0,0,1764,100));//ѡ����������  
	Mat dst;  
	gemm(test,cc,1,0,0,dst,GEMM_2_T);  
	Mat dst1;  
	gemm(dst,cc,1,0,0,dst1);  
	printf("%d. diff = %g\n", 1, norm(test, dst1, NORM_L2));  
	return 0;  
}  */