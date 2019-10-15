#include "mainwindow.h"
#include "ui_mainwindow.h"


#include <QFileDialog>

#include <string>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>


#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "NormalizationLib.h"
#include "DispLib.h"
#include "histograms.h"
#include "gradient.h"
#include "RegionU16Lib.h"

#include "mazdaroi.h"
#include "mazdaroiio.h"

#include <tiffio.h>


using namespace boost;
using namespace std;
using namespace boost::filesystem;
using namespace cv;
//------------------------------------------------------------------------------------------------------------------------------
string MatPropetiesAsText(Mat Im)
{
    string Out ="Image properties: ";
    Out += "max x = " + to_string(Im.cols);
    Out += ", max y = " + to_string(Im.rows);
    Out += ", # channels = " + to_string(Im.channels());
    Out += ", depth = " + to_string(Im.depth());

    switch(Im.depth())
    {
    case CV_8U:
        Out += "CV_8U";
        break;
    case CV_8S:
        Out += "CV_8S";
        break;
    case CV_16U:
        Out += "CV_16U";
        break;
    case CV_16S:
        Out += "CV_16S";
        break;
    case CV_32S:
        Out += "CV_32S";
        break;
    case CV_32F:
        Out += "CV_32F";
        break;
    case CV_64F:
        Out += "CV_64F";
        break;
    default:
        Out += " unrecognized ";
    break;
    }
    return Out;
}
//------------------------------------------------------------------------------------------------------------------------------
string TiffFilePropetiesAsText(string FileName)
{
    float xRes,yRes;
    uint32 imWidth, imLength;
    uint16 resolutionUnit;
    TIFF *tifIm = TIFFOpen(FileName.c_str(),"r");
    string Out ="Tiff properties: ";
    if(tifIm)
    {
        TIFFGetField(tifIm, TIFFTAG_XRESOLUTION , &xRes);
        TIFFGetField(tifIm, TIFFTAG_YRESOLUTION , &yRes);
        TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH , &imWidth);
        TIFFGetField(tifIm, TIFFTAG_IMAGELENGTH , &imLength);
        TIFFGetField(tifIm, TIFFTAG_RESOLUTIONUNIT , &resolutionUnit);


        Out += "max x = " + to_string(imLength);
        Out += ", max y = " + to_string(imWidth);
        Out += ", ResUnit = " + to_string(resolutionUnit);
        Out += ", xRes = " + to_string(1.0/xRes);
        Out += ", yRes = " + to_string(1.0/yRes);
        TIFFClose(tifIm);
    }
    else
        Out += " improper file ";
    //TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH, &width);

    return Out;
}
//------------------------------------------------------------------------------------------------------------------------------

bool GetTiffProperties(string FileName, float &xRes, float &yRes)
{
    //float xRes,yRes;
    //uint32 imWidth, imLength;
    //uint16 resolutionUnit;
    TIFF *tifIm = TIFFOpen(FileName.c_str(),"r");
    string Out ="Tiff properties: ";
    if(tifIm)
    {
        TIFFGetField(tifIm, TIFFTAG_XRESOLUTION , &xRes);
        TIFFGetField(tifIm, TIFFTAG_YRESOLUTION , &yRes);
        //TIFFGetField(tifIm, TIFFTAG_IMAGEWIDTH , &imWidth);
        //TIFFGetField(tifIm, TIFFTAG_IMAGELENGTH , &imLength);
        //TIFFGetField(tifIm, TIFFTAG_RESOLUTIONUNIT , &resolutionUnit);

        TIFFClose(tifIm);
        return 1;
    }
    else
    {
        xRes = 1.0;
        yRes = 1.0;
        return 0;
    }
}
//------------------------------------------------------------------------------------------------------------------------------
cv::Mat MaskBackround(cv::Mat ImIn)
{
    Mat Mask;
    Mask.release();
    if(ImIn.channels() != 3)
    {
        return Mask;
    }

    int maxX = ImIn.cols;
    int maxY = ImIn.rows;
    int maxXY = maxX * maxY;
    Mask = Mat::zeros(maxY, maxX, CV_16U);

    uint16 *wMask = (uint16 *)Mask.data;

    unsigned char *wImIn = (unsigned char *)ImIn.data;

    for(int i = 0; i < maxXY; i++)
    {
        unsigned char B = *wImIn;
        wImIn++;
        unsigned char G = *wImIn;
        wImIn++;
        unsigned char R = *wImIn;
        wImIn++;

        if(R > 95 && G > 40 && B > 20 && R - G > 15 && R > G && R > B)
            *wMask = 1;
        wMask++;
    }

    return Mask;
}

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          constructor Destructor
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->comboBoxOutputMode->addItem("Gray");
    ui->comboBoxOutputMode->addItem("B");
    ui->comboBoxOutputMode->addItem("G");
    ui->comboBoxOutputMode->addItem("R");
    ui->comboBoxOutputMode->setCurrentIndex(0);

    ui->comboBoxGradient->addItem("Morfological");
    ui->comboBoxGradient->addItem("Up");
    ui->comboBoxGradient->addItem("Down");
    ui->comboBoxGradient->setCurrentIndex(0);

}

MainWindow::~MainWindow()
{
    delete ui;
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//          CLASS FUNCTIONS
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::OpenImageFolder()
{
    if (!exists(ImageFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + ImageFolder.string()+ " not exists "));
        ImageFolder = "d:\\";
    }
    if (!is_directory(ImageFolder))
    {
        ui->textEditOut->append(QString::fromStdString(" Image folder : " + ImageFolder.string()+ " This is not a directory path "));
        ImageFolder = "C:\\Data\\";
    }
    ui->lineEditImageFolder->setText(QString::fromStdString(ImageFolder.string()));
    ui->listWidgetImageFiles->clear();
    for (directory_entry& FileToProcess : directory_iterator(ImageFolder))
    {
        regex FilePattern(ui->lineEditRegexImageFile->text().toStdString());
        if (!regex_match(FileToProcess.path().filename().string().c_str(), FilePattern ))
            continue;
        path PathLocal = FileToProcess.path();
        if (!exists(PathLocal))
        {
            ui->textEditOut->append(QString::fromStdString(PathLocal.filename().string() + " File not exists" ));
            break;
        }
        ui->listWidgetImageFiles->addItem(PathLocal.filename().string().c_str());
    }

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ReadImage()
{
    if(ui->checkBoxAutocleanOut->checkState())
        ui->textEditOut->clear();
    int flags;
    if(ui->checkBoxLoadAnydepth->checkState())
        flags = CV_LOAD_IMAGE_ANYDEPTH;
    else
        flags = IMREAD_COLOR;
    ImIn = imread(FileName, flags);
    if(ImIn.empty())
    {
        ui->textEditOut->append("improper file");
        return;
    }

    path FileNamePath(FileName);
    string extension = FileNamePath.extension().string();

    if((extension == ".tif" || extension == ".tiff") && ui->checkBoxShowTiffInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(TiffFilePropetiesAsText(FileName)));

    if(ui->checkBoxShowMatInfo->checkState())
        ui->textEditOut->append(QString::fromStdString(MatPropetiesAsText(ImIn)));

    ProcessImages();



}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowsScaledImage(Mat Im, string ImWindowName)
{
    if(Im.empty())
    {
        ui->textEditOut->append("Empty Image to show");
        return;
    }

    Mat ImToShow;


    ImToShow = Im.clone();

    displayScale = pow(double(ui->spinBoxScaleBase->value()), double(ui->spinBoxScalePower->value()));
    if (displayScale != 1.0)
        cv::resize(ImToShow,ImToShow,Size(), displayScale, displayScale, INTER_AREA);
    if(ui->checkBoxImRotate->checkState())
        rotate(ImToShow,ImToShow, ROTATE_90_CLOCKWISE);
    imshow(ImWindowName, ImToShow);

}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ShowImages()
{
    if(ui->checkBoxShowInput->checkState())
        ShowsScaledImage(ImIn, "Input Image");

    if(ui->checkBoxShowGray->checkState())
        ShowsScaledImage(ShowImage16Gray(ImGray,ui->doubleSpinBoxFixMinDisp->value(),
                                         ui->doubleSpinBoxFixMaxDisp->value()), "Gray Image");

    if(ui->checkBoxShowPC->checkState())
        ShowsScaledImage(ShowImage16PseudoColor(ImGray,ui->doubleSpinBoxFixMinDisp->value(),
                                         ui->doubleSpinBoxFixMaxDisp->value()), "PC Image");

    if(ui->checkBoxShowGradient->checkState() && ui->checkBoxProcessGradient->checkState())
        ShowsScaledImage(ShowImage16PseudoColor(ImGradient,ui->doubleSpinBoxFixMinDispGrad->value(),
                                         ui->doubleSpinBoxFixMaxDispGrad->value()), "Gradient Image");

    if(ui->checkBoxShowMask->checkState())
        ShowsScaledImage(ShowRegion(Mask), "Mask");
}
//------------------------------------------------------------------------------------------------------------------------------
void MainWindow::ProcessImages()
{
    if(ImIn.empty())
    {

        return;
    }
    Mat ImGrayTemp;
    if(ui->checkBoxLoadAnydepth->checkState())
    {
        ImGrayTemp = ImIn;
    }
    else
    {
        switch(ui->comboBoxOutputMode->currentIndex())
        {
            case 1:
            {
                Mat Planes[3];
                split(ImIn,Planes);
                ImGrayTemp = Planes[0];
            }
            break;

            case 2:
            {
                Mat Planes[3];
                split(ImIn,Planes);
                ImGrayTemp = Planes[1];
            }
            break;
            case 3:
            {
                Mat Planes[3];
                split(ImIn,Planes);
                ImGrayTemp = Planes[2];
            }
        break;
        default:
            cvtColor(ImIn, ImGrayTemp, CV_BGR2GRAY);
        break;
        }
    }

    ImGrayTemp.convertTo(ImGray,CV_16U);

    if(ui->checkBoxProcessGradient->checkState())
    {
        switch(ui->comboBoxGradient->currentIndex())
        {
        case 1:
            ImGradient = GradientUP(ImGray);
            ImGradient.convertTo(ImGradient,CV_16U,10);
        break;
        case 2:
            ImGradient = GradientDown(ImGray);
            ImGradient.convertTo(ImGradient,CV_16U,10);
        break;
        default:
            ImGradient = GradientMorph(ImGray,ui->spinBoxGradientSchape->value())*10;
        break;
        }
        Mask = Threshold16(ImGradient, ui->spinBoxGradThreshold->value());

    }
    if(ui->checkBoxMaskBackGround->checkState() && ImIn.channels() == 3)
    {
        Mask = MaskBackround(ImIn);
    }

    ShowImages();
}
//------------------------------------------------------------------------------------------------------------------------------
//          Slots
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

void MainWindow::on_pushButtonOpenImageFolder_clicked()
{
    QFileDialog dialog(this, "Open Folder");
    dialog.setFileMode(QFileDialog::Directory);
    dialog.setDirectory(QString::fromStdString(ImageFolder.string()));

    if(dialog.exec())
    {
        ImageFolder = dialog.directory().path().toStdWString();
    }
    else
        return;
    OpenImageFolder();
}

void MainWindow::on_lineEditRegexImageFile_returnPressed()
{
    OpenImageFolder();
}

void MainWindow::on_listWidgetImageFiles_currentTextChanged(const QString &currentText)
{

    path fileToOpen = ImageFolder;
    fileToOpen.append(currentText.toStdString());
    FileName = fileToOpen.string();
    ReadImage();
    path fileToOpenTxt = ImageFolder;
    fileToOpenTxt.append(fileToOpen.stem().string() +".txt");
    FileNameTxt = fileToOpenTxt.string();
    ui->textEditOut->append(QString::fromStdString(FileNameTxt));
}

void MainWindow::on_checkBoxShowInput_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_checkBoxShowTiffInfo_toggled(bool checked)
{
    ReadImage();
}

void MainWindow::on_checkBoxShowMatInfo_toggled(bool checked)
{
    ReadImage();
}

void MainWindow::on_checkBoxAutocleanOut_stateChanged(int arg1)
{
    ReadImage();
}

void MainWindow::on_checkBoxLoadAnydepth_toggled(bool checked)
{
    ReadImage();
}

void MainWindow::on_checkBoxShowOutput_toggled(bool checked)
{
    ShowImages();
}


void MainWindow::on_spinBoxScalePower_valueChanged(int arg1)
{
    ShowImages();
}

void MainWindow::on_spinBoxScaleBase_valueChanged(int arg1)
{
    ShowImages();
}

void MainWindow::on_checkBoxImRotate_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_pushButtonOpenOutFolder_clicked()
{

}

void MainWindow::on_doubleSpinBoxFixMinDisp_valueChanged(double arg1)
{
    ProcessImages();
}

void MainWindow::on_doubleSpinBoxFixMaxDisp_valueChanged(double arg1)
{
    ProcessImages();
}

void MainWindow::on_checkBoxShowGray_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_checkBoxShowGradient_toggled(bool checked)
{
    ShowImages();
}

void MainWindow::on_doubleSpinBoxFixMinDispGrad_valueChanged(double arg1)
{
    ShowImages();
}

void MainWindow::on_doubleSpinBoxFixMaxDispGrad_valueChanged(double arg1)
{
     ShowImages();
}

void MainWindow::on_spinBoxGradientSchape_valueChanged(int arg1)
{
    ProcessImages();
}

void MainWindow::on_spinBoxGradThreshold_valueChanged(int arg1)
{
    ProcessImages();
}

void MainWindow::on_comboBoxGradient_currentIndexChanged(int index)
{
    ProcessImages();
}

void MainWindow::on_checkBoxShowMask_toggled(bool checked)
{
     ShowImages();
}

void MainWindow::on_comboBoxOutputMode_currentIndexChanged(int index)
{
     ProcessImages();
}

void MainWindow::on_checkBoxShowPC_toggled(bool checked)
{
     ShowImages();
}
