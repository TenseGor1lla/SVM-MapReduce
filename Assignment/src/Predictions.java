import java.io.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;


public class Predictions {
	public static void main(String[] args) throws Exception {
		
	    Configuration conf = new Configuration();
	    Path test=new Path("hdfs:/BDAAssignment/test.txt");
	    Path weightsPath=new Path("hdfs:/BDAAssignment/weights.txt");
	    Path biasPath=new Path("hdfs:/BDAAssignment/bias.txt");
		FileSystem fs=FileSystem.get(conf);
		BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(weightsPath)));
		String[] weights = br.readLine().split(" ");
		double w1 = Double.parseDouble(weights[0]);
		double w2 = Double.parseDouble(weights[1]);
		br.close();
		br=new BufferedReader(new InputStreamReader(fs.open(biasPath)));
		double bias = Double.parseDouble(br.readLine());
		br.close();
		br=new BufferedReader(new InputStreamReader(fs.open(test)));
		int total=0,cnt=0;
		String line = br.readLine();
		while(line!=null) {
			String[] arr = line.split(" ");
			double x1 = Double.parseDouble(arr[0]);
			double x2 = Double.parseDouble(arr[1]);
			int y = Integer.parseInt(arr[2]);
			int ans = predict(x1,x2,w1,w2,bias);
			if(y==ans) cnt++;
 			line = br.readLine();
 			total++;
		}
		System.out.println("Accuracy : "+(double)cnt/total);
	
	}
	
	static int predict(double x1, double x2,double w1, double w2,double bias) {
		double tmp = x1*w1 + x2*w2 + bias;
		if (tmp<=0) return 0;
		else return 1;
	}
}

