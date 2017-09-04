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


#define CENTRAL_CROP true   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体

int main()
{
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人

	char filename[100];	
	char filename1[100];	
	
	for(int num=1; num<10; num++)
	{
		
		sprintf(filename,"F:/研究生/视频/train/%d.png",num);
		
		Mat src = imread(filename);//读取图片

		vector<float> descriptors;//HOG描述子向量

		hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
	
		//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
		if( 1 == num )
		{
			DescriptorDim = descriptors.size();//HOG描述子的维数
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
			sampleFeatureMat = Mat::zeros(20, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(20, 1, CV_32FC1);
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
		sampleLabelMat.at<float>(num,0) = 1;//正样本类别为1，有人
	}

	//依次读取负样本图片，生成HOG描述子
	for(int num=1; num<10 ; num++)
	{
		sprintf(filename1,"F:/研究生/视频/test/%d.png",num);
		Mat src = imread(filename1);//读取图片
		//resize(src,img,Size(64,128));

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
		//cout<<"描述子维数："<<descriptors.size()<<endl;

		for(int i=0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num+10,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
		sampleLabelMat.at<float>(num+10,0) = -1;//负样本类别为-1，无人
	}
	
	

	cout<<DescriptorDim;
	////输出样本的HOG特征向量矩阵到文件
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
	vector<string> img_path;//输入文件名变量       
	ifstream svm_data(filename.c_str());    
	string buf;    
	int len=0;    
	while( svm_data )//将训练样本文件依次读取进来        
	{       

		if( getline( svm_data, buf ) )        
		{       
			//if( len%2==0)    
			img_path.push_back( buf );//图像路径        
		}    
		//len++;    
	}    
	svm_data.close();//关闭文件   
	return img_path;  
}  
int _tmain(int argc, _TCHAR* argv[])  
{  
	vector<string> img_path;//输入文件名变量     
	vector<int> img_catg;      
	int nLine = 0;      
	string buf;      
	locale loc = locale::global(locale(""));  
	ifstream svm_data( "pos.txt" );//训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件  
	locale::global(loc);  

	ofstream zhengyangben_txt;  
	float result;  


	unsigned long n;       
	while( svm_data )//将训练样本文件依次读取进来      
	{      
		if( getline( svm_data, buf ) )      
		{      
			nLine ++;      
			if( nLine % 2 == 0 )//注：奇数行是图片全路径，偶数行是标签   
			{      
				img_catg.push_back( atoi( buf.c_str() ) );//atoi将字符串转换成整型，标志(0,1，2，...，9)，注意这里至少要有两个类别，否则会出错      
			}      
			else      
			{      
				img_path.push_back( buf );//图像路径      
			}      

		}      
	}   
	int num=0;  
	svm_data.close();//关闭文件      
	CvMat *data_mat, *res_mat;      
	int nImgNum = nLine / 2; //nImgNum是样本数量，只有文本行数的一半，另一半是标签       
	data_mat = cvCreateMat( nImgNum, 1764, CV_32FC1 );  //第二个参数，即矩阵的列是由下面的descriptors的大小决定的，可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的    
	cvSetZero( data_mat );      
	//类型矩阵,存储每个样本的类型标志      
	res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );      
	cvSetZero( res_mat );      
	IplImage* src;      
	IplImage* trainImg=cvCreateImage(cvSize(64,64),8,3);//需要分析的图片，这里默认设定图片是64*64大小，所以上面定义了1764，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行      

	//处理HOG特征    
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
		vector<float>descriptors;//存放结果       
		hog->compute(trainImg, descriptors,Size(1,1), Size(0,0)); //Hog特征计算        

		cout<<"HOG dims: "<<descriptors.size()<<endl;          
		n=0;      
		for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)      
		{      
			cvmSet(data_mat,i,n,*iter);//存储HOG特征   
			n++;      
		}         


		if(img_catg[i]==-1)  
			img_catg[i]=2;  
		else  
			num++;  
		cvmSet( res_mat, i, 0, img_catg[i] );      
		// cout<<" 处理完毕: "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;      
	}      

	Mat p_data=Mat(data_mat);  
	Mat mean_=Mat(1,1764,CV_32FC1);  
	Mat mm = Mat(1,num,CV_32FC1);  
	mm=1;  
	gemm(mm,p_data,1,mean_,0,mean_);  
	mean_=mean_/num;  
	PCA pca;  
	pca(p_data,mean_,0,100);//选择特征数量  
	Mat mmm=pca.eigenvalues;  
	Mat test=p_data(Rect(0,0,1764,1));  
	Mat dd = Mat(mmm.size(),CV_32FC1);  
	pca.project(test,dd); //投影  
	Mat rec;  
	pca.backProject(dd,rec); //反向投影，还原  
	printf("%d. diff = %g\n", 1, norm(test, rec, NORM_L2));  

	SVD svd;  
	svd(p_data);  
	Mat v=svd.vt;  

	Mat cc=v(Rect(0,0,1764,100));//选择特征数量  
	Mat dst;  
	gemm(test,cc,1,0,0,dst,GEMM_2_T);  
	Mat dst1;  
	gemm(dst,cc,1,0,0,dst1);  
	printf("%d. diff = %g\n", 1, norm(test, dst1, NORM_L2));  
	return 0;  
}  */